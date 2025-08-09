# Multi Objective COverage Bayesian Optimization (MOCOBO)
Official implemention of the Multi Objective COverage Bayesian Optimization (MOCOBO) method we proposed in our paper titled "Covering Multiple Objectives with a Small Set of Solutions Using Bayesian Optimization" (https://arxiv.org/abs/2501.19342).  
This repository includes code to run the MOCOBO method on all multi-objective optimization tasks from the paper. 

## Weights and Biases (wandb) tracking
This repo it set up to automatically track optimization progress using the Weights and Biases (wandb) API. Wandb stores and updates data during optimization and automatically generates live plots of progress. If you are unfamiliar with wandb, we recommend creating a free account here:
https://wandb.ai/site

Then, for all runs of MOCOBO using the scripts/run_mocobo.py, use the argument --wandb_entity to specify the username for your personal wandb account. Optimization progress will then be automatically logged to your account. 

## Getting Started

### Cloning the Repo (Git Lfs)
This repository uses git lfs to store larger data files and model checkpoints. Git lfs must therefore be installed before cloning the repository. 

```Bash
conda install -c conda-forge git-lfs
```

### Docker
To set up the environment we used to run all tasks for this paper, we recommend using docker. 
You can use the public docker image nmaus/opt (https://hub.docker.com/r/nmaus/opt), or build it yourself using docker/Dockerfile.

The image tone mapping task has additional environment requirements that conflict with other tasks. To run MOCOBO for the image tone mapping tasks, instead use the docker image nmaus/optimg (https://hub.docker.com/r/nmaus/optimg). 

## How to Run MOCOBO
In this section we provide commands that can be used to start a MOCOBO optimization after the environment has been set up. To start an MOCOBO run, run scripts/run_mocobo.py with desired command line arguments. 

Remember to also add the --wandb_entity argument to specify the username for your personal wandb account. 

### Args
To get a list of all available command line arguments with default values and descriptions of each, run the following:

```Bash
python3 run_mocobo.py -- --help
```

### Tasks 
Use the argument --task_id to specify the multi-objective optimization task you would like to optimize with MOCOBO. Each task has a specific identifying string $TASK_ID. For example, you can run MOCOBO on a specific task with id $TASK_ID using the following command: 

```Bash
python3 run_mocobo.py --task_id $TASK_ID - run - done 
```

This code base provides support for the following multi-objective optimization tasks:

| task_id | Multi-Objective Optimization Task | Number of Objectives for Task (T) | Number of Covering Solutions (K) Used In the Paper |
|---------|--------------------|-------------------|-----------|
|  rover     | Rover, navigating multiple obstacle courses    | 4, 8, or 12 (specified with "rover_n_obstacle_courses" arg) | K = 2, 2, 4 when T = 4, 8, 12 respectively |
|  apex      | Peptide design with the APEX Oracle  | 11 | K = 4 |
|  rano     | Molecule design, adding new elements to Ranolazine | 6 | K = 3 |
|  img     | Image tone mapping, optimizing multiple quality metrics | 7 | K = 4 |

### Task specific arguments 

1. img_task_version

For the image tone mapping task, use the "img_task_version" argument to specify which version of the task to run (which initial hdr image to use). Use --img_task_version 1 to specify the church hdr benchmark image and use --img_task_version 2 to specify the desk hdr benchmark image. 

2. rover_n_obstacle_courses

For the rover task, use the "rover_n_obstacle_courses" argument to specify the number of obstacle cousres the rover must naviatge (equivalently the total number of objectives we seek to cover with MOCOBO). This codebase supports rover_n_obstacle_courses 4, 8, and 12. 

3. apex_constraint

For apex task, use the boolean "apex_constraint" argument to specify whether to apply the template constrained described in the paper. If apex_constraint is set to True, the optimized peptides are constrained to be at least 75 percent similar to the closest template extinct organism amino acid sequence. If apex_constraint is set to False, this constraint is not applied. 

### Example Commands for Each Task

#### Rover Task  

With four obstacle courses: 
```Bash
python3 run_mocobo.py --task_id rover --k 2 --rover_n_obstacle_courses 4 --max_n_oracle_calls 80000 - run - done 
```
With eight obstacle courses: 
```Bash
python3 run_mocobo.py --task_id rover --k 2 --rover_n_obstacle_courses 8 --max_n_oracle_calls 100000 - run - done 
```
With 12 obstacle courses: 
```Bash
python3 run_mocobo.py --task_id rover --k 4 --rover_n_obstacle_courses 12 --max_n_oracle_calls 100000 - run - done 
```

#### Apex Task (Antimicrobial Peptide Design)

Template free (no template constraint):

```Bash
python3 run_mocobo.py --task_id apex --k 4 --max_n_oracle_calls 600000 --apex_constraint False --update_e2e True - run - done 
```

Template constrained (with extinct peptide template similarity constraint):

```Bash
python3 run_mocobo.py --task_id apex --k 4 --max_n_oracle_calls 2000000 --apex_constraint True --update_e2e True - run - done 
```

#### Molecule Design Task (Adding New Elements to Ranolazine)

```Bash
python3 run_mocobo.py --task_id rano --k 3 --max_n_oracle_calls 2000000 --update_e2e True - run - done 
```

#### Image Tone Mapping Task 

Version 1 with church HDR image: 
```Bash
python3 run_mocobo.py --task_id img --k 4 --max_n_oracle_calls 70000 --img_task_version 1 - run - done 
```

Version 2 with desk HDR image: 
```Bash
python3 run_mocobo.py --task_id img --k 4 --max_n_oracle_calls 70000 --img_task_version 2 - run - done 
```
