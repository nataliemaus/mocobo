import torch 
import sys 
sys.path.append("../")
from tasks.objective import Objective
from tasks.utils.rover_utils import ConstantOffsetFn, create_multiple_domains_mocobo


class RoverMultipleObstacleCourses(Objective):
    ''' Rover optimization task
        Goal is to find a policy for the Rover which
        results in a trajectory that moves the rover from
        start point to end point while avoiding the obstacles,
        thereby maximizing reward 
        Converted to a MO task by desinging solutions that allow the 
        Rover to nagivate multiple domains (multiple obstacle courses)
    ''' 
    def __init__(
        self,
        n_obstacle_courses,
        dim=60,
        f_max=5.0,
        **kwargs,
    ):
        assert dim % 2 == 0
        lb = -0.5 * 4 / dim 
        ub = 4 / dim 

        assert n_obstacle_courses in [3,4,8,12] # only set up for 4,8, or 12
        self.domains_list = create_multiple_domains_mocobo(
            n_domains=n_obstacle_courses,
            n_points=dim // 2,
        )
        self.oracles = []
        for domain in self.domains_list:
            oracle_t = ConstantOffsetFn(domain, f_max)
            self.oracles.append(oracle_t)

        super().__init__(
            dim=dim,
            lb=lb,
            ub=ub,
            **kwargs,
        ) 

    def f(self, x):
        self.num_calls += 1
        ys = []
        for oracle in self.oracles:
            ys.append(oracle(x.cpu().numpy()))

        ys = torch.tensor(ys).to(dtype=self.dtype) # (T,)
        ys = ys.unsqueeze(0) # (1,T) = (1,n_domains)
        return ys 

# TODO: delete below, remove pngs: 
if __name__ == "__main__":
    from tasks.utils.plot_rover import plot_rover
    obj = RoverMultipleObstacleCourses(
        n_obstacle_courses=3,
    )
    for i, domain in enumerate(obj.domains_list):
        x = torch.randn(obj.dim)
        save_plot_path = f"test_rover_domain_plots/ablation_all_conflict_{i+1}.png"
        plot_rover(domain, x, save_plot_path, plot_title=None, trajectory_color=None, linewidth=None)
