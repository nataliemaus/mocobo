import torch 

def get_random_init_data(num_initialization_points, objective):
    ''' randomly initialize num_initialization_points
        total initial data points to kick-off optimization 
        Returns the following:
            init_train_x (a tensor of x's)
            init_train_y (a tensor of scores/y's)
    '''
    lb, ub = objective.lb, objective.ub 
    if lb is None:
        # if no bounds, just take random normal sample 
        init_train_x = torch.randn(num_initialization_points, objective.dim)
    else:
        init_train_x = torch.rand(num_initialization_points, objective.dim)*(ub - lb) + lb
    return_dict = objective(init_train_x)
    init_train_y = return_dict["ys"]
    init_train_strings = return_dict["strings"]

    return init_train_x, init_train_y, init_train_strings
