import wandb 

def create_wandb_tracker(wandb_project_name, wandb_entity, config_dict):
    tracker = wandb.init(
        project=wandb_project_name,
        entity=wandb_entity,
        config=config_dict,
    ) 
    wandb_run_name = wandb.run.name 

    return tracker, wandb_run_name