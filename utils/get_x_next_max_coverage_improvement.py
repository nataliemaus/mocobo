import torch
import time 
import sys 
sys.path.append("../")
from utils.get_best_covering_set import get_coverage_score, algo_1_greedy_approxmation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_x_next_max_coverage_improvement(
    models_list,
    X_cand, # (N_cand, dim)
    y_obs, # (N_obs, T)
    current_best_coverage_score, # float 
    k=4,
    avg_over_n_samples=1, # S
    batch_size=20,
):
    N_cand = X_cand.shape[0]
    T = y_obs.shape[1]
    # print("Starting getting x_next from X_cand", X_cand.shape)

    try:    
        all_y_samples = []  # compresed y samples
        for model in models_list:
            posterior = model.posterior(X_cand, observation_noise=False)
            y_samples_t = posterior.rsample(sample_shape=torch.Size([avg_over_n_samples]))  
            all_y_samples.append(y_samples_t)
        all_y_samples = torch.cat(
            all_y_samples, dim=-1
        )  # (avg_over_n_samples, N_cand, T) 
    except: # TODO: remove try/except!! (only necessary for no tr ablation for APEX!!)
        all_y_samples = torch.zeros(avg_over_n_samples, N_cand, T) + y_obs.min().item()
    
    all_y_samples = all_y_samples.to(device=device)
    assert all_y_samples.shape[0] == avg_over_n_samples
    assert all_y_samples.shape[1] == N_cand
    assert all_y_samples.shape[2] == T 

    avg_coverage_improvements = []
    for cand_ix in range(N_cand):
        cand_coverage_improvements = []
        for sample_ix in range(avg_over_n_samples):
            single_sample = all_y_samples[sample_ix][cand_ix].unsqueeze(0) # (1,T)
            coverage_imporvement_s = approximate_coverage_improvement(
                y_sample=single_sample, 
                y_obs=y_obs,
                current_best_coverage_score=current_best_coverage_score,
                k=k,
            )
            cand_coverage_improvements.append(coverage_imporvement_s)
        
        avg_coverage_improvements.append(torch.tensor(cand_coverage_improvements).mean().item())

    avg_coverage_improvements = torch.tensor(avg_coverage_improvements) # (N_cand,)
    best_cand_idcs = torch.topk(avg_coverage_improvements, k=batch_size).indices
    x_next = X_cand[best_cand_idcs]

    return x_next 


def approximate_coverage_improvement(
    y_sample, 
    y_obs, 
    current_best_coverage_score, 
    k, 
):
    ''' 
    returns max(0, coverage_imporvement) (non-negative)
    where coverage_imporvement is improvement in coverage score by adding y_sample to y_obs 
    '''
    y_obs = y_obs.to(device)
    new_y = torch.cat((y_sample, y_obs),0) # (N_obs+1, T)
    A_star = algo_1_greedy_approxmation(y=new_y, k=k,)
    new_approx_selected_indicies = list(A_star)
    # if the new datapoint wasn't selected for best covering set, no imporvement possible 
    if not (0 in new_approx_selected_indicies):
        return 0.0 
    new_best_val = get_coverage_score(new_y[new_approx_selected_indicies])
    coverage_improvement = new_best_val - current_best_coverage_score 
    coverage_improvement = max(0.0, coverage_improvement)

    return coverage_improvement
