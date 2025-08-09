import sys 
sys.path.append("../")
import torch
import numpy as np
import fire 
import pandas as pd
import warnings
import copy
warnings.filterwarnings('ignore')
import os
os.environ["WANDB_SILENT"] = "True"
import signal 
import gpytorch
from gpytorch.mlls import PredictiveLogLikelihood
from utils.gp_shared_dkl import GPModelSharedDKL 
from utils.simple_nn import DenseNetwork
from utils.generate_candidates import generate_batch
from utils.train_model import (
    update_gp_models, 
    update_single_model,
    update_models_end_to_end_with_vae,
)
from utils.create_wandb_tracker import create_wandb_tracker
from utils.set_seed import set_seed 
from utils.get_random_init_data import get_random_init_data
from utils.turbo import TurboState, update_state
from utils.get_best_covering_set import update_best_covering_set
from tasks.utils.apex_template_similarity_constraint import get_perc_similarity_to_closest_tempalte
from constants import (
    SAVE_DATA_DIRECTORY,
    PATH_TO_APEX_INITIALIZATION_SEQS,
    get_apex_init_best_covering_set_path,
    PATH_TO_APEX_INITIALIZATION_YS,
    PATH_TO_APEX_INITIALIZATION_ZS,
    PATH_TO_APEX_INITIALIZATION_CVALS,
    PATH_TO_RANO_INITIALIZATION_SEQS,
    PATH_TO_RANO_INITIALIZATION_YS,
    PATH_TO_RANO_INITIALIZATION_ZS,
    PATH_TO_IMG_INITIALIZATION_YS,
    PATH_TO_IMG_INITIALIZATION_XS,
)
# for specific tasks
from tasks.rover import RoverMultipleObstacleCourses
try:
    from tasks.rano_guacamol_objective import RanoGuacamolObjective
    SUCCESSFUL_RANO_IMPORT = True 
except:
    print("\nFAILED TO IMPORT RANOLAZINE OBJECTIVE IN THIS ENV\n")
    SUCCESSFUL_RANO_IMPORT = False 

try:
    from tasks.img_objective import ImgObjective
    SUCCESSFUL_IMG_IMPORT = True 
except:
    print("\nFAILED TO IMPORT IMG OBJECTIVE IN THIS ENV\n")
    SUCCESSFUL_IMG_IMPORT = False 

from tasks.apexgo_objective import ApexGoObjective
task_id_to_objective = {}
task_id_to_objective['apex'] = ApexGoObjective
if SUCCESSFUL_RANO_IMPORT: # works in Docker image opt:latest 
    task_id_to_objective['rano'] = RanoGuacamolObjective
if SUCCESSFUL_IMG_IMPORT: # works in Docker image optimg:latest 
    task_id_to_objective['img'] = ImgObjective
task_id_to_objective['rover'] = RoverMultipleObstacleCourses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Optimize(object):
    """
    Run Multi Objective COverage Bayesian Optimization (MOCOBO)
    Args:
        task_id: String id for optimization task in task_id_to_objective dict 
        seed: Random seed to be set. If None, no particular random seed is set
        wandb_entity: Username for your wandb account for wandb logging
        wandb_project_name: Name of wandb project where results will be logged, if none specified, will use default f"mocobo-{task_id}"
        k: int giving the number of points we seek to form our best covering set. 
        max_n_oracle_calls: Max number of oracle calls allowed (budget). Optimization run terminates when this budget is exceeded
        bsz: Acquisition batch size
        train_bsz: batch size used for model training/updates
        num_initialization_points: Number of initial random data points used to kick off optimization
        lr: Learning rate for model updates
        n_update_epochs: Number of epochs to update the model for on each optimization step
        n_inducing_pts: Number of inducing points for GP
        grad_clip: clip the gradeint at this value during model training 
        normalize_ys: If True, normalize objective values for training (recommended, typical when using GP models)
        max_allowed_n_failures_improve_loss: We train model until the loss fails to improve for this many epochs
        train_to_convergence: if true, train until loss stops improving instead of only for n_update_epochs
        max_allowed_n_epochs: When we train to convergence, we also cap the number of epochs to this max allowed value
        update_on_n_pts: Update model on this many data points on each iteration.
        float_dtype_as_int: specify integer either 32 or 64, dictates whether to use torch.float32 or torch.float64 
        max_n_brute_fore_comp: Max number to pass into brute force computation of best covering set (Larger --> slower but more accurate approximation)
        n_acquisition_candidates: n candidates to sample from trust region during acquisition 
        avg_over_n_samples: n samples from model to average over when estimating coverage improvement 
        apex_constraint: if True, apply constraint to be at least i.e. 75% similar to one of the template sequences for apex task
        apex_sim_threshold: float giving the minimum perc similarity to closest tempalte sequence we require for apex constraint 
        avg_over_n_samples_c_model: n samples to average over when getting constraint models' average pred cval for candidate
        update_e2e: update GP end to end with VAE for latent-space BO tasks (LOL-BO)
        e2e_freq: number of concecutive failures to make progress between each end-to-end update 
        n_e2e_update_epochs: number of epochs to update vae and models end-to-end 
        e2e_lr: learning rate for end to end updates (lower for more stable vae+gp updates)
        verbose: if True, print optimization progress updates 
        verbose2: if True, print lots of extra update statements, use for debugging 
        run_id: Optional string run id. Only use is for wandb logging to identify a specific run
        img_task_version: int in [1,2]. Specifies version of image tone mapping task. 1-->chruch, 2-->desk 
        rover_n_obstacle_courses: int in [3,4,8,12], specifies number of obstacles courses (equivalently number of objectives) for rover task
    """
    def __init__(
        self,
        task_id: str="apex",
        seed: int=None,
        wandb_entity: str="",
        wandb_project_name: str="",
        k: int=4,
        max_n_oracle_calls: int=20_000,
        bsz: int=20,
        train_bsz: int=256,
        num_initialization_points: int=2_000,
        lr: float=0.001,
        n_update_epochs: int=5,
        n_inducing_pts: int=1024,
        grad_clip: float=1.0,
        normalize_ys=True,
        max_allowed_n_failures_improve_loss: int=3,
        train_to_convergence=True,
        max_allowed_n_epochs: int=30, 
        update_on_n_pts: int=1_000,
        float_dtype_as_int: int=32,
        max_n_brute_fore_comp: int=100,
        n_acquisition_candidates: int=500,
        avg_over_n_samples: int=1,
        apex_constraint=False,
        apex_sim_threshold: float=0.75,
        avg_over_n_samples_c_model: int=20,
        update_e2e=False,
        e2e_freq: int=10, 
        n_e2e_update_epochs: int=2,
        e2e_lr: float=0.0001,
        verbose=True,
        verbose2=True,
        run_id: str="",
        img_task_version=1,
        rover_n_obstacle_courses=4,
        no_trust_region_ablation=False,
        standard_ei_ablation=False,
        ei_num_restarts=10, # arg 1 for ei ablation 
        ei_raw_samples=256, # arg 2 for ei ablation 
    ):
        if float_dtype_as_int == 32:
            self.dtype = torch.float32
            torch.set_default_dtype(torch.float32)
        elif float_dtype_as_int == 64:
            self.dtype = torch.float64
            torch.set_default_dtype(torch.float64)
        else:
            assert 0, f"float_dtype_as_int must be one of: [32, 64], instead got {float_dtype_as_int}"
        
        # log all args to wandb
        self.method_args = {}
        self.method_args['init'] = locals()
        del self.method_args['init']['self']
        wandb_config_dict = {k: v for method_dict in self.method_args.values() for k, v in method_dict.items()}

        assert img_task_version in [1,2] 
        assert rover_n_obstacle_courses in [3,4,8,12] 
        self.img_task_version = img_task_version
        self.rover_n_obstacle_courses = rover_n_obstacle_courses
        self.e2e_lr = e2e_lr
        self.n_e2e_update_epochs = n_e2e_update_epochs
        self.e2e_freq = e2e_freq
        self.update_e2e = update_e2e
        self.apex_sim_threshold = apex_sim_threshold
        self.avg_over_n_samples_c_model = avg_over_n_samples_c_model
        self.apex_constraint = apex_constraint
        self.avg_over_n_samples = avg_over_n_samples
        self.n_acquisition_candidates = n_acquisition_candidates
        self.max_n_brute_fore_comp = max_n_brute_fore_comp
        self.k = k 
        self.run_id = run_id
        self.task_id = task_id 
        self.init_training_complete = False
        self.normalize_ys = normalize_ys
        self.update_on_n_pts = update_on_n_pts
        self.verbose = verbose
        self.verbose2 = verbose2
        self.max_allowed_n_failures_improve_loss = max_allowed_n_failures_improve_loss
        self.max_allowed_n_epochs = max_allowed_n_epochs
        self.max_n_oracle_calls = max_n_oracle_calls
        self.n_inducing_pts = n_inducing_pts
        self.lr = lr
        self.n_update_epochs = n_update_epochs
        self.train_bsz = train_bsz
        self.grad_clip = grad_clip
        self.bsz = bsz
        self.train_to_convergence = train_to_convergence
        self.num_initialization_points = num_initialization_points
        self.no_trust_region_ablation = no_trust_region_ablation
        self.standard_ei_ablation = standard_ei_ablation
        self.ei_num_restarts = ei_num_restarts 
        self.ei_raw_samples = ei_raw_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(seed)

        # start wandb tracker
        if not wandb_project_name:
            wandb_project_name = f"mocobo-{task_id}"
        # tracker, wandb_run_name
        self.tracker, self.wandb_run_name = create_wandb_tracker(
            wandb_project_name=wandb_project_name,
            wandb_entity=wandb_entity,
            config_dict=wandb_config_dict,
        )
        signal.signal(signal.SIGINT, self.handler)

        # initialize objective 
        if task_id == "img":
            self.objective = task_id_to_objective["img"](
                dtype=self.dtype, 
                version=img_task_version,
            )
        elif task_id == "rover":
            self.objective = task_id_to_objective["rover"](
                dtype=self.dtype, 
                n_obstacle_courses=rover_n_obstacle_courses,
            )
        else:
            self.objective = task_id_to_objective[task_id](dtype=self.dtype)

        # get initialization data 
        if self.verbose2:
            print("getting init data")
        self.get_initialization_data()
        if self.verbose2:
            print("getting normed train ys")
        self.train_y_mean, self.train_y_std, self.normed_train_y = self.initialize_normed_train_ys()

        # None indicicates they must be initialized still 
        self.tr_states = None 
 
        # get inducing points
        if self.verbose2:
            print("getting inducing points")
        inducing_points = self.get_initial_inducing_pts()
        if self.verbose2:
            print("initializing gp models")
        # Define approximate GP models (one gp per output dim, shared deep kernel)
        shared_feature_extractor = DenseNetwork(
            input_dim=self.train_x.size(-1), hidden_dims=(self.objective.dim, self.objective.dim),
        ).to(device)
        self.gp_models = []
        self.gp_mlls = []
        for _ in range(self.train_y.shape[-1]):# one gp for each output dim (T output dims)
            model = GPModelSharedDKL(
                inducing_points=inducing_points.to(self.device),
                likelihood=gpytorch.likelihoods.GaussianLikelihood(),
                shared_feature_extractor=shared_feature_extractor,
            ).to(device)
            self.gp_models.append(model)

            mll = PredictiveLogLikelihood(
                model.likelihood, model, num_data=self.train_x.shape[0]
            )
            self.gp_mlls.append(mll)

        # add model for constraint function if using 75% similarity constraint for apex 
        if self.verbose2:
            print("initializing constraint models (if applicable)")
        self.constraint_model = None 
        self.constraint_mll = None 
        if self.apex_constraint: 
            assert self.task_id == "apex"
            constraint_feature_extractor = DenseNetwork(
                input_dim=self.all_train_x_w_infeasible.size(-1), hidden_dims=(self.objective.dim, self.objective.dim),
            ).to(device)
            self.constraint_model = GPModelSharedDKL( 
                inducing_points=inducing_points.to(self.device),
                likelihood=gpytorch.likelihoods.GaussianLikelihood(),
                shared_feature_extractor=constraint_feature_extractor,
            ).to(device)
            self.constraint_mll = PredictiveLogLikelihood(
                self.constraint_model.likelihood, self.constraint_model, num_data=self.all_train_x_w_infeasible.shape[0],
            )
        if self.verbose2:
            print("finished init")


    def load_presaved_init_best_coverage(self,):
        best_indexes, best_coverage_score = None, None 
        if (self.task_id == "apex") and (not self.apex_constraint):
            # load init best covering set pre-computed with closer approximation 
            # only pre-computed for unconstrained case (when all init data is kept)
            path_to_init_covering_set = get_apex_init_best_covering_set_path(k=self.k)
            df = pd.read_csv(path_to_init_covering_set)
            best_coverage_score = df["best_coverage_score"].values[0].item()
            best_indexes = []
            for i in range(self.k):
                best_idx = df[f"best_index_{i}"].values[0].item()
                best_indexes.append(best_idx) 

        return best_indexes, best_coverage_score
    
    def get_initialization_data(self,):
        # None unless we are applying a constraint: 
        train_c = None 
        all_train_c_w_infeasible = None 
        all_train_x_w_infeasible = None 
        # Define objective and get initialization data 
        if self.task_id == "apex": # initialize with pre-computed 20k apex dataset 
            # load train seqs (strings)
            df = pd.read_csv(PATH_TO_APEX_INITIALIZATION_SEQS)
            train_strings = df["seqs"].values.tolist()
            # load train_ys (MICs)
            ys = np.load(PATH_TO_APEX_INITIALIZATION_YS) # 20k,11
            ys = torch.from_numpy(ys).to(dtype=self.dtype)
            train_y = ys*-1 # convert minimization problem to maximization problem 
            # load train_xs (latent zs)
            train_x = torch.load(PATH_TO_APEX_INITIALIZATION_ZS)
            train_x = train_x.to(dtype=self.dtype)
            # Constraint values: 
            if self.apex_constraint:
                c_df = pd.read_csv(PATH_TO_APEX_INITIALIZATION_CVALS)
                train_c = c_df["max_perc_similarity"].values 
                # all c vals used to train constraint model even if they don't meet constraints
                train_c = torch.from_numpy(train_c).to(dtype=self.dtype) 
                # all cs and xs for trianing c model 
                all_train_c_w_infeasible = train_c 
                all_train_x_w_infeasible = train_x 
                # Remove data NOT meeting constraint:  
                meet_constraint_bool_array = train_c >= self.apex_sim_threshold
                train_y = train_y[meet_constraint_bool_array]
                train_x = train_x[meet_constraint_bool_array]
                train_c = train_c[meet_constraint_bool_array]
                train_strings = np.array(train_strings)[meet_constraint_bool_array.numpy()]
                train_strings = train_strings.tolist()
            self.num_initialization_points = train_y.shape[0]
        elif self.task_id == "rano":
            # load train seqs (strings)
            df = pd.read_csv(PATH_TO_RANO_INITIALIZATION_SEQS)
            train_strings = df["smile"].values.tolist()
            # load train_ys 
            train_y = np.load(PATH_TO_RANO_INITIALIZATION_YS) # (10k,T)
            train_y = torch.from_numpy(train_y).to(dtype=self.dtype)
            assert train_y.shape[1] == 6 # T=6 for this task 
            # load train_xs (latent zs)
            train_x = torch.load(PATH_TO_RANO_INITIALIZATION_ZS)
            train_x = train_x.to(dtype=self.dtype)
            # make sure cut off init zs at 10k to match size of other init data
            self.num_initialization_points = train_y.shape[0]
            train_strings = train_strings[0:self.num_initialization_points]
            train_x = train_x[0:self.num_initialization_points]
        elif self.task_id == "img": 
            xs_path = PATH_TO_IMG_INITIALIZATION_XS
            ys_path = PATH_TO_IMG_INITIALIZATION_YS
            img_version = self.objective.version
            if img_version > 1:
                xs_path = xs_path.replace(".npy", f"_v{img_version}.npy")
                ys_path = ys_path.replace(".npy", f"_v{img_version}.npy")
            if self.verbose:
                print(f"\n\n\n Loading init img data from paths:\n{xs_path}\n{ys_path} \n\n\n")
            train_x = np.load(xs_path) # (2000, 13)
            train_y = np.load(ys_path) # (2000, 8)
            self.num_initialization_points = train_x.shape[0]
            assert self.num_initialization_points == 2000 
            assert train_y.shape[0] == self.num_initialization_points
            assert train_x.shape[1] == 13 # dim=13
            assert train_y.shape[1] == 7 # T 
            train_x = torch.from_numpy(train_x).to(dtype=self.dtype)
            train_y = torch.from_numpy(train_y).to(dtype=self.dtype)
            train_strings = None 
        else: 
            # get random init training data 
            train_x, train_y, train_strings = get_random_init_data(
                num_initialization_points=self.num_initialization_points,
                objective=self.objective,
            )

        if self.verbose:
            print("train x shape:", train_x.shape)
            print("train y shape:", train_y.shape)
            if train_strings is not None:
                print(f"N={len(train_strings)} train strings, i.e.", train_strings[0:2])
        
        self.train_x = train_x
        self.train_y = train_y 
        self.train_c = train_c 
        self.all_train_c_w_infeasible = all_train_c_w_infeasible
        self.all_train_x_w_infeasible = all_train_x_w_infeasible
        self.train_strings = train_strings 
        return self 


    def initialize_normed_train_ys(self,):
        train_y_mean = self.train_y.mean()
        train_y_std = self.train_y.std()
        if train_y_std == 0:
            train_y_std = 1
        if self.normalize_ys:
            normed_train_y = (self.train_y - train_y_mean) / train_y_std
        else:
            normed_train_y = self.train_y
        
        return train_y_mean, train_y_std, normed_train_y


    def get_initial_inducing_pts(self,):
        if len(self.train_x) >= self.n_inducing_pts:
            inducing_points = self.train_x[0:self.n_inducing_pts,:]
        else:
            lb, ub = self.objective.lb, self.objective.ub
            if (lb is None): # if no absolute bounds 
                lb = self.train_x.min().item() 
                ub = self.train_x.max().item()
                temp_range_bounds = ub - lb 
                lb = lb - temp_range_bounds*0.1 # add small buffer lower than lowest we've seen 
                ub = ub + temp_range_bounds*0.1 # add small buffer higher than highest we've seen 
            n_extra_ind_pts = self.n_inducing_pts - len(self.train_x)
            extra_ind_pts = torch.rand(n_extra_ind_pts, self.objective.dim)*(ub - lb) + lb
            inducing_points = torch.cat((self.train_x, extra_ind_pts), -2)
        
        return inducing_points 


    def get_top_indicies_per_task(self,):
        T = self.train_y.shape[-1]
        top_per_t = self.update_on_n_pts//T
        top_per_t = min(top_per_t, self.train_y.shape[0])
        top_indicies_all = []
        for t_ in range(T):
            top_indicies_t = torch.topk(self.train_y[:,t_], k=top_per_t).indices
            top_indicies_all = top_indicies_all + top_indicies_t.tolist() 
        top_indicies_all = list(set(top_indicies_all))
        return top_indicies_all 

    def grab_data_for_update(self,best_indexes):
        if not self.init_training_complete:
            x_update_on = self.train_x
            normed_y_update_on = self.normed_train_y.squeeze()
            c_update_on = self.train_c
            c_update_on_w_infeasilbe = self.all_train_c_w_infeasible
            x_update_on_w_infeasilbe = self.all_train_x_w_infeasible
            self.init_training_complete = True
            strings_update_on = self.train_strings 
        else:
            # update on latest collected update_on_n_pts data, plus best covering set k seen so far 
            total_n_data = self.train_x.shape[0]
            top_indicies_per_task = self.get_top_indicies_per_task()
            recent_batch_size = int(self.bsz*self.k) # one batch of points tr 
            latest_data_idxs = np.arange(total_n_data)[-recent_batch_size:].tolist()
            idxs_update_on = best_indexes + latest_data_idxs + top_indicies_per_task 
            idxs_update_on = list(set(idxs_update_on)) # removes duplicates 
            x_update_on = self.train_x[idxs_update_on]
            normed_y_update_on = self.normed_train_y.squeeze()[idxs_update_on]
            if self.train_c is None:
                c_update_on = None 
            else:
                c_update_on = self.train_c[idxs_update_on]

            if self.train_strings is None:
                strings_update_on = None 
            else:
                strings_update_on = np.array(self.train_strings)[idxs_update_on].tolist()

            if self.all_train_c_w_infeasible is None:
                c_update_on_w_infeasilbe = None 
                x_update_on_w_infeasilbe = None 
            else:
                # best indexes don't apply to full train c/x with infeasilbe data included 
                c_update_on_w_infeasilbe = self.all_train_c_w_infeasible[-recent_batch_size:]
                x_update_on_w_infeasilbe = self.all_train_x_w_infeasible[-recent_batch_size:]

        self.x_update_on = x_update_on
        self.normed_y_update_on = normed_y_update_on
        self.c_update_on = c_update_on
        self.c_update_on_w_infeasilbe = c_update_on_w_infeasilbe
        self.x_update_on_w_infeasilbe = x_update_on_w_infeasilbe
        self.strings_update_on = strings_update_on

        return self 

    def initialize_save_data_dir(self,):
        save_data_dir = f"{SAVE_DATA_DIRECTORY}/{self.task_id}/"
        if not os.path.exists(save_data_dir):
            os.mkdir(save_data_dir)
        save_data_dir = save_data_dir + f"{self.wandb_run_name}/"
        if not os.path.exists(save_data_dir):
            os.mkdir(save_data_dir)
        self.save_data_dir = save_data_dir
        return self 


    def save_run_data(self, best_cov_score, best_indexes):
        ''' Save best covering set of K solutions found so far locally 
            Organize data in a CSV giving the reward obtained for each task by each solution 
        ''' 
        if self.train_strings is None:
            # if no string representation for task, save best k solutions (best_k_xs) directly
            best_k_xs = self.train_x[best_indexes] # (k,D) = K solutions x Dimension of search space
            np.save(f"{self.save_data_dir}best_k_solutions.npy", best_k_xs.numpy())

        best_k_ys = self.train_y[best_indexes] # (k,T), rewards of best k solutions on each task 
        solution_nums = np.arange(self.k) + 1
        results_df = {
            "overall_coverage_score":[best_cov_score]*self.k,
            "solution_num":solution_nums,
        }
        if self.train_strings is not None:
            # list of k solutions as strings 
            results_df["solution"] = [self.train_strings[idx] for idx in best_indexes] 
        for t_ in range(self.train_y.shape[-1]): # loop over T objectives
            key_t = f"objective_{t_+1}_reward"
            results_df[key_t] = [] # rewards per solution for objective t
            for k_ in range(self.k): 
                reward = best_k_ys[k_,t_].item()
                results_df[key_t].append(reward)

        results_df = pd.DataFrame.from_dict(results_df)
        results_df.to_csv(f"{self.save_data_dir}results_for_best_covering_set_found.csv", index=False)

        return self 


    def run(self):
        ''' Main optimization loop
        '''
        self.initialize_save_data_dir()
        self.total_n_infeasible_thrown_out = 0 # count num candidates thrown out due to not meeting constraint(s)
        prev_best_coverage_score = -torch.inf 
        self.progress_fails_since_last_e2e = 0 # for LOL-BO
        self.count_n_e2e_updates = 0 
        self.count_n_e2e_lr_reductions = 0
        self.count_n_e2e_update_failures = 0
        self.from_which_tr = [None]*len(self.train_y) # track which tr each data point came from 
        bo_loop_num = 0 
        if self.verbose2:
            print("loading presaved init best coverage")
        best_indexes, best_coverage_score = self.load_presaved_init_best_coverage()
        if self.verbose2:
            print("starting bo loop")
        while self.objective.num_calls < self.max_n_oracle_calls:
            bo_loop_num += 1 
            if self.verbose2:
                print(f"BO Loop num: {bo_loop_num}, N calls so far: {self.objective.num_calls}")  
                print(f"Updating best covering set")
            # compute best covering set 
            best_indexes, best_coverage_score = update_best_covering_set(
                train_y=self.train_y, 
                k=self.k, 
                current_best_indicies=best_indexes, 
                current_best_value=best_coverage_score, 
                max_n_brute_fore=self.max_n_brute_fore_comp,
            )
            self.best_xs = self.train_x[best_indexes] 
            if self.train_c is not None:
                self.best_cs = self.train_c[best_indexes]
            self.best_from_which_tr = [self.from_which_tr[bidx] for bidx in best_indexes]
            if best_coverage_score > prev_best_coverage_score:
                self.successful_step = True 
                if self.verbose2:
                    print(f"coverage imporved, saving data")
                prev_best_coverage_score = best_coverage_score
                # if we improved, save results (new best covering set) locally 
                self.save_run_data(best_cov_score=best_coverage_score, best_indexes=best_indexes) 
            else: 
                self.successful_step = False 
                if self.verbose2:
                    print(f"coverage NOT imporved, incrementing progress_fails_since_last_e2e")
                # if no imporvement, count one failure to imporve 
                self.progress_fails_since_last_e2e += 1 
            
            # Print progress update and update wandb with optimization progress
            n_calls_ = self.objective.num_calls
            if self.verbose:
                print(f"Task: {self.task_id}, Wandb run: {self.wandb_run_name}, BO Loop num: {bo_loop_num}, After {n_calls_} oracle calls, best coverage score = {best_coverage_score}")
            dict_log = {
                "best_coverage_score":best_coverage_score,
                "n_oracle_calls":n_calls_,
                "total_n_infeasible_thrown_out":self.total_n_infeasible_thrown_out,
                "count_n_e2e_updates":self.count_n_e2e_updates,
                "count_n_e2e_lr_reductions":self.count_n_e2e_lr_reductions,
                "count_n_e2e_update_failures":self.count_n_e2e_update_failures,
            }
            if self.verbose2:
                print(f"Logging data to wandb")
            self.tracker.log(dict_log)

            # Normalize train ys
            if self.verbose2:
                print(f"Normalizing ys")
            if self.normalize_ys:
                self.normed_train_y = (self.train_y - self.train_y_mean) / self.train_y_std
            else:
                self.normed_train_y = self.train_y
            
            # Update model on data 
            if self.verbose2:
                print(f"Grab data for update")
            self.grab_data_for_update(best_indexes)
            if self.verbose2:
                print(f"Update model(s) on data")
            if (self.update_e2e and (self.progress_fails_since_last_e2e >= self.e2e_freq)):
                # Do end-to-end vae+gp updates (LOL-BO)
                self.update_surrogate_models_and_vae_end_to_end() 
            else: 
                # otherwise, just update the surrogate models on data
                self.update_surrogate_models()

            # update trust region state 
            if self.verbose2:
                print(f"Update tr states")
            self.tr_states = self.update_trust_regions()

            # Generate a batch of candidates 
            if self.verbose2:
                print(f"get x next")
            x_next = [] 
            from_which_tr_next = []
            for tr_index, tr_i in enumerate(self.tr_states):
                x_next_i = generate_batch(
                    gp_models_list=self.gp_models,
                    X=self.train_x,  
                    batch_size=self.bsz,
                    y_obs=self.train_y, # (N_obs, T)
                    k=self.k, # int
                    current_best_coverage_score=best_coverage_score, # float
                    tr_center=tr_i.center,
                    tr_length=tr_i.length,
                    dtype=self.dtype,
                    device=self.device,
                    lb=self.objective.lb,
                    ub=self.objective.ub,
                    n_candidates=self.n_acquisition_candidates,
                    avg_over_n_samples=self.avg_over_n_samples,
                    c_model=self.constraint_model,
                    apex_sim_threshold=self.apex_sim_threshold,
                    apex_constraint=self.apex_constraint,
                    avg_over_n_samples_c_model=self.avg_over_n_samples_c_model,
                    standard_ei_ablation=self.standard_ei_ablation,
                    no_trust_region_ablation=self.no_trust_region_ablation,
                    ei_num_restarts=self.ei_num_restarts, 
                    ei_raw_samples=self.ei_raw_samples,
                ) 
                x_next.append(x_next_i)
                from_which_tr_next = from_which_tr_next + [tr_index]*len(x_next_i)
            x_next = torch.cat(x_next, 0) # (bsz*k, dim)
            assert x_next.shape[-1] == self.objective.dim 
            assert len(from_which_tr_next) == len(x_next) # which tr did each cand come from 

            # Evaluate candidates
            if self.verbose2:
                print(f"Evaluating candidates")
            out_dict = self.objective(x_next)
            y_next = out_dict["ys"]
            strings_next = out_dict["strings"]  
            if self.apex_constraint:
                # Compute constraint values used to remove new data that doesn't meet constraint 
                c_next = get_perc_similarity_to_closest_tempalte(
                    peptide_seqs_list=strings_next,
                    dtype=self.dtype,
                )
            else:
                c_next = None 
            if self.verbose2:
                print(f"Update data collected with xnext ynext")
            self.update_datasets(
                x_next=x_next, 
                y_next=y_next, 
                c_next=c_next, 
                strings_next=strings_next, 
                from_which_tr_next=from_which_tr_next,
            )

        # terminate wandb tracker 
        self.tracker.finish()
        return self


    def update_datasets(self, x_next, y_next, c_next, strings_next, from_which_tr_next):
        if self.apex_constraint:
            # Remove new data that doesn't meet constraint 
            self.all_train_c_w_infeasible = torch.cat((self.all_train_c_w_infeasible, c_next))
            self.all_train_x_w_infeasible = torch.cat((self.all_train_x_w_infeasible, x_next), dim=-2)
            feasible_next_bool_array = c_next >= self.apex_sim_threshold
            # Subtract num infeasible from total num oracle calls bc we don't actually need to eval oracle on these 
            num_infeasible = (feasible_next_bool_array == False).sum().item() 
            self.objective.num_calls = self.objective.num_calls - num_infeasible 
            self.total_n_infeasible_thrown_out += num_infeasible
            # update next feasible candidates only 
            x_next = x_next[feasible_next_bool_array] 
            y_next = y_next[feasible_next_bool_array]
            c_next = c_next[feasible_next_bool_array]
            strings_next = np.array(strings_next)[feasible_next_bool_array.numpy()].tolist()
            from_which_tr_next = np.array(from_which_tr_next)[feasible_next_bool_array.numpy()].tolist()

        if strings_next is not None:
            # Remove empty strings 
            good_strings_bool_array = torch.tensor([len(string_n) for string_n in strings_next]) > 0
            num_empty_strings = (good_strings_bool_array == False).sum().item() 
            self.objective.num_calls = self.objective.num_calls - num_empty_strings
            self.total_n_infeasible_thrown_out += num_empty_strings
            # update next feasible candidates only (no empty strings)
            x_next = x_next[good_strings_bool_array] 
            y_next = y_next[good_strings_bool_array]
            if c_next is not None:
                c_next = c_next[good_strings_bool_array]
            strings_next = np.array(strings_next)[good_strings_bool_array.numpy()].tolist()
            from_which_tr_next = np.array(from_which_tr_next)[good_strings_bool_array.numpy()].tolist()

        # Update data
        self.train_x = torch.cat((self.train_x, x_next), dim=-2)
        self.train_y = torch.cat((self.train_y, y_next), dim=-2)
        if self.train_c is not None:
            self.train_c = torch.cat((self.train_c, c_next))
        if strings_next is not None:
            self.train_strings = self.train_strings + strings_next 
        self.from_which_tr = self.from_which_tr + from_which_tr_next
        
        return self 


    def update_trust_regions(self,):
        k = self.best_xs.shape[0]
        updated_tr_states = [] 
        if self.tr_states is None: # if TRS not yet initialized 
            for k_ix in range(k):
                tr_state_k = TurboState(
                    dim=self.train_x.shape[-1],
                    batch_size=self.bsz,
                    center=self.best_xs[k_ix],
                )
                updated_tr_states.append(tr_state_k)
        else:
            for k_ix in range(k):
                # Update TR states 
                # count a success if coverage score improved AND includes a candidate from this TR
                if self.successful_step and (k_ix in self.best_from_which_tr):
                    self.tr_states[k_ix].success_counter += 1
                    self.tr_states[k_ix].failure_counter = 0
                else:
                    self.tr_states[k_ix].success_counter = 0
                    self.tr_states[k_ix].failure_counter += 1
                # update tr length accordingly
                tr_state_k = update_state(
                    state=self.tr_states[k_ix],
                )
                # restart tr if length updated went below threshold min and triggered a restart 
                if tr_state_k.restart_triggered:
                    tr_state_k = TurboState(
                        dim=self.train_x.shape[-1],
                        batch_size=self.bsz,
                        center=self.best_xs[k_ix],
                    )
                # trs always recentered on best covering set
                tr_state_k.center = self.best_xs[k_ix]
                updated_tr_states.append(tr_state_k)
        return updated_tr_states


    def update_surrogate_models_and_vae_end_to_end(self,):
        # includes recentering after e2e updates 
        gp_model_states_before_update = []
        gp_mll_states_before_update = []
        for ix, gp_model_i in enumerate(self.gp_models):
            state_dict_i = copy.deepcopy(gp_model_i.state_dict())
            gp_model_states_before_update.append(state_dict_i)
            mll_state_i = copy.deepcopy(self.gp_mlls[ix].state_dict())
            gp_mll_states_before_update.append(mll_state_i)
        if self.constraint_model is not None:
            c_model_state_pre_update = copy.deepcopy(self.constraint_model.state_dict())
            c_mll_state_pre_update = copy.deepcopy(self.constraint_mll.state_dict())
        try:
            out_dict = update_models_end_to_end_with_vae(
                objective=self.objective, # lsbo objective with vae object 
                gp_models=self.gp_models,
                gp_mlls=self.gp_mlls,
                train_strings=self.strings_update_on,
                train_y=self.normed_y_update_on,
                train_c=self.c_update_on,
                c_model=self.constraint_model,
                c_mll=self.constraint_mll,
                constraint_func=get_perc_similarity_to_closest_tempalte,
                lr=self.e2e_lr,
                n_epochs=self.n_e2e_update_epochs,
                train_bsz=self.train_bsz,
                grad_clip=self.grad_clip,
                dtype=self.dtype,
            )
            # update models 
            self.gp_models = out_dict["gp_models"] 
            self.gp_mlls = out_dict["gp_mlls"] 
            self.constraint_model = out_dict["c_model"] 
            self.constraint_mll = out_dict["c_mll"]  
            self.objective = out_dict["objective"] 
            # update datasets of recentered data 
            self.update_datasets(
                x_next=out_dict["new_xs"], 
                y_next=out_dict["new_ys"], 
                c_next=out_dict["new_cs"] , 
                strings_next=out_dict["new_seqs"],
                from_which_tr_next=[None]*len(out_dict["new_ys"]),
            )
            # reset progress fails since last e2e update to 0
            self.progress_fails_since_last_e2e = 0 
            # count e2e update 
            self.count_n_e2e_updates += 1
        except:
            # NOTE: this except case never occurred during any MOCOBO paper experiments, 
            #       but we leave implementation here just in case 
            # In case of nan loss due to unstable trianing, 
            # 1. reset models to prev state dicts before unstable training 
            for i in range(len(self.gp_models)):
                self.gp_models[i].load_state_dict(gp_model_states_before_update[i])
                self.gp_mlls[i].load_state_dict(gp_mll_states_before_update[i])
            if self.constraint_model is not None:
                self.constraint_model.load_state_dict(c_model_state_pre_update)
                self.constraint_mll.load_state_dict(c_mll_state_pre_update)
            # 2. reduce lr for next time to reduce chance of training instability
            if self.e2e_lr > 1e-06: # don't drop below 1e-06
                self.e2e_lr = self.e2e_lr/2 
                self.count_n_e2e_lr_reductions += 1
            self.count_n_e2e_update_failures += 1

        return self 

    def update_surrogate_models(self,):
        # Update constraint model on all xs and constraint values (including infeasible data that doesn't meet constraints)
        if self.c_update_on_w_infeasilbe is not None:
            self.constraint_model, self.constraint_mll = update_single_model(
                model=self.constraint_model,
                mll=self.constraint_mll,
                train_x=self.x_update_on_w_infeasilbe,
                train_y=self.c_update_on_w_infeasilbe,
                lr=self.lr,
                n_epochs=self.n_update_epochs,
                train_bsz=self.train_bsz,
                grad_clip=self.grad_clip,
                train_to_convergence=self.train_to_convergence, 
                max_allowed_n_failures_improve_loss=self.max_allowed_n_failures_improve_loss,
                max_allowed_n_epochs=self.max_allowed_n_epochs, 
            )
        update_dict = update_gp_models(
            gp_models=self.gp_models,
            gp_mlls=self.gp_mlls,
            train_x=self.x_update_on,
            train_y=self.normed_y_update_on,
            lr=self.lr,
            n_epochs=self.n_update_epochs,
            train_bsz=self.train_bsz,
            grad_clip=self.grad_clip,
            train_to_convergence=self.train_to_convergence, 
            max_allowed_n_failures_improve_loss=self.max_allowed_n_failures_improve_loss,
            max_allowed_n_epochs=self.max_allowed_n_epochs, 
        )
        self.gp_models = update_dict["gp_models"]
        self.gp_mlls = update_dict["gp_mlls"]
        return self 
    

    def handler(self, signum, frame):
        # if we Ctrl-c, make sure we terminate wandb tracker
        print("Ctrl-c hass been pressed, terminating wandb tracker...")
        self.tracker.finish()
        msg = "tracker terminated, now exiting..."
        print(msg, end="", flush=True)
        exit(1)
        return None 

    def done(self):
        return None


def new(**kwargs):
    return Optimize(**kwargs)


if __name__ == "__main__":
    fire.Fire(Optimize)