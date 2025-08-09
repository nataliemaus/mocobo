import sys 
sys.path.append("../")
import torch
from torch.quasirandom import SobolEngine
from utils.get_x_next_max_coverage_improvement import get_x_next_max_coverage_improvement
try:
    from botorch.acquisition import qExpectedImprovement
    from botorch.optim import optimize_acqf
except:
    print("Failed to import qExpectedImprovement, optimize_acqf needed for standard EI baseline")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_batch(
    gp_models_list,
    X,
    batch_size,
    y_obs, # (N_obs, T)
    k, # int 
    current_best_coverage_score, # float
    tr_center,
    tr_length,
    n_candidates=None,
    dtype=torch.float32,
    device=torch.device('cuda'),
    avg_over_n_samples=1,
    lb=None,
    ub=None,
    # constraint args (SCBO)
    c_model=None,
    apex_sim_threshold=0.75,
    apex_constraint=False,
    avg_over_n_samples_c_model=20,
    standard_ei_ablation=False, 
    no_trust_region_ablation=False,
    ei_num_restarts=10, 
    ei_raw_samples=256,
):
    dim = X.shape[-1]
    if n_candidates is None: 
        n_candidates = min(5000, max(2000, 200 * X.shape[-1])) 
    if (lb is None): # if no absolute bounds 
        lb = X.min().item() 
        ub = X.max().item()
        temp_range_bounds = ub - lb 
        lb = lb - temp_range_bounds*0.1 # add small buffer lower than lowest we've seen 
        ub = ub + temp_range_bounds*0.1 # add small buffer higher than highest we've seen 

    sobol = SobolEngine(dim, scramble=True) 
    sobol_cands = sobol.draw(n_candidates).to(dtype=dtype).to(device)
    if no_trust_region_ablation:
        # get sobol candidates in bounds problem domain 
        X_cand = lb + (ub - lb) * sobol_cands
    else:
        # get candidates in trust region (turbo implementation)
        weights = torch.ones_like(tr_center) * (ub - lb)
        # get upper an lower bounds of trust region 
        tr_lb = torch.clamp(tr_center - weights * tr_length / 2.0, lb, ub).to(device)
        tr_ub = torch.clamp(tr_center + weights * tr_length / 2.0, lb, ub).to(device)
        # get sobol candidates in bounds trust region 
        pert = tr_lb + (tr_ub - tr_lb) * sobol_cands
        # Create a perturbation mask 
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (torch.rand(n_candidates, dim, dtype=dtype, device=device)<= prob_perturb)
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1
        mask = mask.to(device)
        # Create candidate points from the perturbations and the mask
        X_cand = tr_center.expand(n_candidates, dim).clone()
        X_cand = X_cand.to(device)
        X_cand[mask] = pert[mask]
    
    # if constrained OPT with SCBO, remove x cands that c_model predicts will not meet constraint 
    if apex_constraint:
        with torch.no_grad():
            X_cand = remove_samples_unlikely_to_meet_constraint(
                c_model=c_model,
                X_cand=X_cand, # (N_cand, dim)
                avg_over_n_samples=avg_over_n_samples_c_model, # S
                apex_sim_threshold=apex_sim_threshold,
                batch_size=batch_size,
            )
        if X_cand.shape[0] == batch_size: # if we have exactly bsz candidates that meet constraints, we are done 
            return X_cand.detach().cpu()
    
    # Ablate baseline acquisition function of using standard EI for each of the T objectives. 
    if standard_ei_ablation:
        assert not no_trust_region_ablation, "running both ablations at once not supported"
        # for each objective 
        x_next_list = []
        for t_, gp_t in enumerate(gp_models_list):
            # select batch by maximizing EI for this objective only 
            y_max_t = y_obs[:,t_].max().item()
            ei = qExpectedImprovement(
                model=gp_t.cuda(), 
                best_f=y_max_t, # .cuda(),
            ) 
            # optimize_acqf returns: batch_candidates, batch_acq_values
            x_next_t, _ = optimize_acqf( 
                acq_function=ei, 
                bounds=torch.stack([tr_lb, tr_ub]).cuda(), # (2,dim)
                q=batch_size, 
                num_restarts=ei_num_restarts, 
                raw_samples=ei_raw_samples, 
            ) # (bsz,dim) 
            x_next_list.append(x_next_t)
        X_next = torch.cat(x_next_list, 0) # (bsz*T,dim)
    else:
        # ECI (our approach)
        with torch.no_grad():
            X_next = get_x_next_max_coverage_improvement(
                models_list=gp_models_list,
                X_cand=X_cand.to(device), # (N_cand, dim)
                y_obs=y_obs, # (N_obs, T)
                current_best_coverage_score=current_best_coverage_score, # float 
                k=k,
                avg_over_n_samples=avg_over_n_samples,
                batch_size=batch_size,
            ) # (bsz, dim)

    return X_next.detach().cpu()


def remove_samples_unlikely_to_meet_constraint(
    c_model,
    X_cand, # (N_cand, dim)
    avg_over_n_samples=1, # S
    apex_sim_threshold=0.75,
    batch_size=20,
):
    N_cand = X_cand.shape[0]
    dim = X_cand.shape[1]
    all_y_samples = []  # compresed y samples
    posterior = c_model.posterior(X_cand, observation_noise=False)
    all_y_samples = posterior.rsample(sample_shape=torch.Size([avg_over_n_samples])) # (avg_over_n_samples, N_cand, 1)
    all_y_samples = all_y_samples.squeeze() # (avg_over_n_samples, N_cand) i.e. torch.Size([10, 500])

    # average over S samples to get avg c_val per candidate 
    avg_pred_c_val = all_y_samples.mean(0) # (N_cand,)
    assert avg_pred_c_val.shape[0] == N_cand
    assert len(avg_pred_c_val.shape) == 1 
    feasible_x_cand = X_cand[avg_pred_c_val >= apex_sim_threshold] # (N_feasible, dim)
    # if none/ too few of the samples meet the constraints, pick the batch_size candidates that minimize violation (SCBO paper)
    if feasible_x_cand.shape[0] < batch_size:
        # equivalently, take top k predicted perc edit distances from templates
        min_violator_indicies = torch.topk(avg_pred_c_val, k=batch_size).indices # torch.Size([batch_size]) of ints 
        feasible_x_cand = X_cand[min_violator_indicies] # (batch_size, dim)
        assert feasible_x_cand.shape[0] == batch_size
    assert feasible_x_cand.shape[1] == dim 
    assert len(feasible_x_cand.shape) == 2

    # return only X cands that are likley to be feasible according to c_model 
    return feasible_x_cand # (N_feasible, dim)
