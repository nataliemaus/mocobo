import torch 
import pandas as pd 
import numpy as np 
import os 
import glob 
import sys 
sys.path.append("../") 
from tasks.rover import RoverMultipleObstacleCourses
from tasks.utils.plot_rover import plot_rover
from constants import SAVE_DATA_DIRECTORY
import argparse 


# different color for each trajectory in set of k solutions
trajectory_k_to_color_dict = { 
    1:"magenta",
    2:"blue",
    3:"darkviolet",
    4:"darkorange",
}

def plot_rover_trajectories_from_best_k_solutions(
    file_type="png",
):  
    ''' 
    Script for plotting the rover trajectories for the best covering set of 
    K solutions found by all completed MOCOBO runs, 
    trajectories are plotted on top of each of the T obstacle courses 
    '''
    best_k_solutions_filename = "best_k_solutions.npy"
    # grab best solutions for all completed runs
    all_completed_run_data_dirs = glob.glob(f"{SAVE_DATA_DIRECTORY}/rover/*/{best_k_solutions_filename}")
    all_completed_run_data_dirs = [path1.replace(best_k_solutions_filename, "") for path1 in all_completed_run_data_dirs]

    for run_data_dir in all_completed_run_data_dirs:
        wandb_run_name = run_data_dir.split("/")[-2]
        print(f"\nPlotting Rover Trajectories From Wandb Run: {wandb_run_name}")
        save_plots_dir = f"{run_data_dir}trajectory_plots/"
        if not os.path.exists(save_plots_dir):
            os.mkdir(save_plots_dir)
        
        run_results_df = pd.read_csv(f"{run_data_dir}results_for_best_covering_set_found.csv")
        T = 0 
        for key in run_results_df.keys():
            if "_reward" in key:
                T += 1 
        obj = RoverMultipleObstacleCourses(n_obstacle_courses=T)
        
        best_k_solutions = np.load(f"{run_data_dir}{best_k_solutions_filename}") # (K,D)
        best_k_solutions = torch.from_numpy(best_k_solutions).float()
        for k_, solution in enumerate(best_k_solutions):
            for t_, domain in enumerate(obj.domains_list): 
                save_plot_path = f"{save_plots_dir}obstacle_course_{t_+1}_solution_{k_+1}.{file_type}"
                plot_title = f"Wandb Run ID:{wandb_run_name}, Obstacle Course {t_+1}, Solution {k_+1} Trajectory" 
                plot_rover(
                    domain=domain, 
                    x=solution, 
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
    plot_rover_trajectories_from_best_k_solutions(file_type=args.file_type)

