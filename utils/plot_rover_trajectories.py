import torch 
import pandas as pd 
import numpy as np 
import argparse 
import os 
import glob 
import sys 
sys.path.append("../") 
from tasks.rover import RoverMultipleObstacleCourses
from tasks.utils.plot_rover import plot_rover
from constants import SAVE_DATA_DIRECTORY


# different color for each trajectory in set of k solutions
trajectory_k_to_color_dict = { 
    1:"magenta",
    2:"blue",
    3:"darkviolet",
    4:"darkorange",
    5:"cyan", 
    6:"lime",
    7:"saddlebrown",
    8:"darkolivegreen",
    9:"deeppink",
}

def analyze_rover(
    eps=1e-3, 
    file_type="png",
):
    save_data_dirs = glob.glob(f"{SAVE_DATA_DIRECTORY}/rover/*/results_for_best_covering_set_found.csv")
    save_data_dirs = [path1.replace("results_for_best_covering_set_found.csv", "") for path1 in save_data_dirs]
    for save_data_dir in save_data_dirs:
        dir_wandb_run_name = save_data_dir.split("/")[-2]
        print(f"\nFor Wandb Run: {dir_wandb_run_name}")
        # best coverage score and indexes for each step 
        df = pd.read_csv(f"{save_data_dir}results_for_best_covering_set_found.csv")
        T = df.shape[-1] - 2 
        obj = RoverMultipleObstacleCourses(n_obstacle_courses=T)
        best_cov_score = df["overall_coverage_score"][0].item()
        best_k_ys = df.values[:,2:] # (k,T)
        check_best_cov_score = best_k_ys.max(0).sum().item() 
        best_k_xs = np.load(f"{save_data_dir}best_k_solutions.npy") # (k,d) 
        best_k_xs = torch.from_numpy(best_k_xs).float()
        assert (abs(best_cov_score - check_best_cov_score) < eps)
        
        # make plots of best only also 
        save_plots_dir1 = f"{SAVE_DATA_DIRECTORY}/rover/{dir_wandb_run_name}/best_only/"
        if not os.path.exists(save_plots_dir1):
            os.mkdir(save_plots_dir1)

        for t_id in range(T): # for each objective, get best reward: 
            k_ys = best_k_ys[:,t_id]
            best_idx_objective_t = k_ys.argmax().item() # int in 0,1,2,.. k-1
            best_reward = k_ys[best_idx_objective_t].item()
            best_x = best_k_xs[best_idx_objective_t] # (d,)
            save_plot_path = f"{save_plots_dir1}domain{t_id+1}_best_traj_{best_idx_objective_t+1}.{file_type}"
            if file_type == "pdf":
                plot_title = None
            else:
                plot_title = f"{dir_wandb_run_name}, Domain {t_id+1}, Best Reward={best_reward:.3f} from Trajectory {best_idx_objective_t+1}\nFinal Best Coverage Score:{check_best_cov_score:.3f}" 
            plot_rover(
                domain=obj.domains_list[t_id], 
                x=best_x, 
                save_plot_path=save_plot_path, 
                plot_title=plot_title, 
                trajectory_color=trajectory_k_to_color_dict[best_idx_objective_t+1],
            )

        # Also plot all to get a sense of how individual final trajectories are specialized (great for some domains, bad for others) 
        save_plots_dir = f"{SAVE_DATA_DIRECTORY}/rover/{dir_wandb_run_name}/"
        for k_, best_x_i in enumerate(best_k_xs): 
            for t_, domain in enumerate(obj.domains_list): 
                save_plot_path = f"{save_plots_dir}domain{t_+1}_traj_{k_+1}.{file_type}"
                reward = best_k_ys[k_,t_].item()
                if file_type == "pdf":
                    plot_title = None
                else:
                    plot_title = f"{dir_wandb_run_name}, Domain {t_+1}, Reward for X_{k_+1}={reward:.3f}\nFinal Best Coverage Score:{check_best_cov_score:.3f}" 
                plot_rover(
                    domain=domain, 
                    x=best_x_i, 
                    save_plot_path=save_plot_path, 
                    plot_title=plot_title, 
                    trajectory_color=trajectory_k_to_color_dict[k_+1],
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        "--file_type",
        help=" string for file type for plots, png or pdf",
        type=str,
        default="png",
        required=False,
    ) 
    args = parser.parse_args() 
    analyze_rover(file_type=args.file_type)

