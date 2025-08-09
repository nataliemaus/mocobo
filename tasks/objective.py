# Parent class for differnet objectives/tasks
import numpy as np
import torch


class Objective:
    """Base class for any optimization task
    class supports oracle calls and tracks
    the total number of oracle class made during
    optimization
    """

    def __init__(
        self,
        dim,
        num_calls=0,
        dtype=torch.float32,
        lb=None,
        ub=None,
    ):
        # track total number of times the oracle has been called
        self.num_calls = num_calls
        # search space dim 
        self.dim = dim 
        # absolute upper and lower bounds on search space
        self.lb = lb
        self.ub = ub
        self.dtype = dtype


    def __call__(self, xs):
        """Function defines batched function f(x) (the function we want to optimize).

        Args:
            xs (enumerable): (bsz, dim) enumerable tye of length equal to batch size (bsz), 
                each item in enumerable type must be a float tensor of shape (dim,) (each is a vector in input search space).

        Returns:
            tensor: (bsz, T) float tensor giving T mo rewards obtained by passing each x in xs into f(x).

        """
        if type(xs) is np.ndarray:
            xs = torch.from_numpy(xs).to(dtype=self.dtype)
        ys = []
        for x in xs:
            ys_i = self.f(x) # (1,T)
            ys.append(ys_i)
        ys = torch.cat(ys, 0) # (bsz, T)
        ys = ys.to(dtype=self.dtype)
        return_dict = {
            "ys":ys,
            "strings":None, # for lsbo type objectives with both strings and rewards outputs 
        }
        return return_dict


    def f(self, x):
        """Function f defines function f(x) we want to optimize.
           This method should also increment self.num_calls by one.

        Args:
            x (tensor): (1, dim) float tensor giving vector in input search space.

        Returns:
            ys (tensor): (1,T) giving the T reward values obtained by evaluating f at x 

        """
        raise NotImplementedError(
            "Must implement f() specific to desired optimization task"
        )
