import itertools
import numpy as np
import scipy.interpolate as si
import itertools 
import os 

class Trajectory:
    def __init__(self):
        pass

    def set_params(self, start, goal, params):
        raise NotImplementedError

    def get_points(self, t):
        raise NotImplementedError

    @property
    def param_size(self):
        raise NotImplementedError


class PointBSpline(Trajectory):
    """
    dim : number of dimensions of the state space
    num_points : number of internal points used to represent the trajectory.
                    Note, internal points are not necessarily on the trajectory.
    """

    def __init__(self, dim, num_points):
        self.tck = None
        self.d = dim
        self.npoints = num_points

    """
    Set fit the parameters of the spline from a set of points. If values are given for start or goal,
    the start or endpoint of the trajectory will be forces on those points, respectively.
    """

    def set_params(self, params, start, goal=None):
        assert start is not None

        points = np.hstack((start[:, None], params.reshape((-1, self.d)).T)).cumsum(
            axis=1
        )
        xp = points[0]
        yp = points[1]
        okay = np.where(np.abs(np.diff(xp)) + np.abs(np.diff(yp)) > 0)
        xp = np.r_[xp[okay], xp[-1]]
        yp = np.r_[yp[okay], yp[-1]] 
        self.tck, u = si.splprep([xp, yp], k=3, s=0)

    def get_points(self, t):
        assert (
            self.tck is not None
        ), "Parameters have to be set with set_params() before points can be queried."
        return np.vstack(si.splev(t, self.tck)).T

    @property
    def param_size(self):
        return self.d * self.npoints


class RoverDomain:
    """
    Rover domain defined on R^d
    cost_fn : vectorized function giving a scalar cost to states
    start : a start state for the rover
    goal : a goal state
    traj : a parameterized trajectory object offering an interface
            to interpolate point on the trajectory
    s_range : the min and max of the state with s_range[0] in R^d are
                the mins and s_range[1] in R^d are the maxs
    """

    def __init__(
        self,
        cost_fn,
        start,
        goal,
        traj,
        force_start=True,
        force_goal=False,
        rnd_stream=None,
    ):
        self.cost_fn = cost_fn
        self.start = start
        self.goal = goal
        self.traj = traj
        self.force_start = force_start
        self.force_goal = force_goal
        self.rnd_stream = rnd_stream

        if self.rnd_stream is None:
            self.rnd_stream = np.random.RandomState(np.random.randint(0, 2 ** 32 - 1))

    # return the negative cost which need to be optimized
    def __call__(self, params, n_samples=1000):
        self.set_params(params)
        return -1 * self.estimate_cost(n_samples=n_samples)

    def set_params(self, params):
        self.traj.set_params(
            params,
            self.start if self.force_start else None,
            self.goal if self.force_goal else None,
        )

    def estimate_cost(self, n_samples=1000):
        # get points on the trajectory
        points = self.traj.get_points(np.linspace(0, 1.0, n_samples, endpoint=True))

        # compute cost at each point
        costs = self.cost_fn(points)

        # estimate (trapezoidal) the integral of the cost along traj 
        avg_cost = 0.5 * (costs[:-1] + costs[1:])
        l = np.linalg.norm(points[1:] - points[:-1], axis=1)
        total_cost = np.sum(l * avg_cost)

        assert self.force_start 
        if not self.force_goal:
            total_cost += 100 * np.linalg.norm(points[-1] - self.goal, 1)
        return total_cost 

    def trajectory(self, params, n_samples=1000):
        self.set_params(params)
        return self.traj.get_points(np.linspace(0, 1.0, n_samples, endpoint=True))
    
    def trajectory_length(self, params, n_samples=1000):
        # Compute the length of the trajectory
        self.set_params(params)
        points = self.traj.get_points(np.linspace(0, 1.0, n_samples, endpoint=True))
        dists = np.sqrt(((points[1:, :] - points[:-1, :]) ** 2).sum(-1))
        trajectory_length = dists.sum()
        return trajectory_length

    def distance_from_goal(self, params, n_samples=1000):
        self.set_params(params)
        points = self.traj.get_points(np.linspace(0, 1.0, n_samples, endpoint=True))
        return np.linalg.norm(points[-1] - self.goal, 1)

    @property
    def input_size(self):
        return self.traj.param_size


class AABoxes:
    def __init__(self, lows, highs):
        self.l = lows
        self.h = highs

    def contains(self, X):
        if X.ndim == 1:
            X = X[None, :]

        lX = self.l.T[None, :, :] <= X[:, :, None]
        hX = self.h.T[None, :, :] > X[:, :, None]

        return lX.all(axis=1) & hX.all(axis=1)


class NegGeom:
    def __init__(self, geometry):
        self.geom = geometry

    def contains(self, X):
        return ~self.geom.contains(X)


class UnionGeom:
    def __init__(self, geometries):
        self.geoms = geometries

    def contains(self, X):
        return np.any(
            np.hstack([g.contains(X) for g in self.geoms]), axis=1, keepdims=True
        )
        

class ConstObstacleCost:
    def __init__(self, geometry, cost):
        self.geom = geometry
        self.c = cost

    def __call__(self, X):
        return self.c * self.geom.contains(X)


class ConstCost:
    def __init__(self, cost):
        self.c = cost

    def __call__(self, X):
        if X.ndim == 1:
            X = X[None, :]
        return np.ones((X.shape[0], 1)) * self.c


class AdditiveCosts:
    def __init__(self, fns):
        self.fns = fns

    def __call__(self, X):
        return np.sum(np.hstack([f(X) for f in self.fns]), axis=1)


class ConstantOffsetFn:
    def __init__(self, fn_instance, offset):
        self.fn_instance = fn_instance
        self.offset = offset

    def __call__(self, x):
        return self.fn_instance(x) + self.offset

    def get_range(self):
        return self.fn_instance.get_range()



def create_cost_function(c, obstacle_delta):
    l = c - obstacle_delta / 2
    h = c + obstacle_delta / 2

    r_box = np.array([[0.5, 0.5]])
    r_l = r_box - 0.5
    r_h = r_box + 0.5

    trees = AABoxes(l, h)
    r_box = NegGeom(AABoxes(r_l, r_h))
    obstacles = UnionGeom([trees, r_box])

    costs = [ConstObstacleCost(obstacles, cost=20.0), ConstCost(0.05)]
    cost_fn = AdditiveCosts(costs)
    return cost_fn, l  


def create_cost_large_four_domains():
    cost_fn_list = []
    obstacle_l_list = []
    obstacle_delta_list = []
    obstacles_arrays_c = []

    # Obstacles Domain 1: 
    a = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    c = np.array(list(itertools.product(a, a)))  
    c = c[1:-1] 
    obstacle_delta = 0.13
    obstacles_arrays_c.append(c)
    obstacle_delta_list.append(obstacle_delta)

    # Obstacles Domain 2: 
    a = [0.1, 0.3, 0.5, 0.7, 0.9]
    c = np.array(list(itertools.product(a, a)))  
    c = c[1:-1] 
    obstacle_delta = 0.13
    obstacles_arrays_c.append(c)
    obstacle_delta_list.append(obstacle_delta)

    # Obstacles Domain 3: 
    c = [
        [0.1, 0.7],
        [0.7, 0.1],
        [0.3, 0.7],
        [0.7, 0.3],
        [0.1, 0.9],
        [0.9, 0.1],
        [0.3, 0.9],
        [0.9, 0.3],
        [0.1, 0.5],
        [0.5, 0.1],
        [0.9, 0.5],
        [0.5, 0.9],
    ]
    c = np.array(c)
    obstacle_delta = 0.2
    obstacles_arrays_c.append(c)
    obstacle_delta_list.append(obstacle_delta)

    # Obstacles Domain 4: 
    c = [
        [0.3, 0.3],
        [0.5, 0.5],
        [0.7, 0.7],
        [0.1, 0.9],
        [0.9, 0.1],
        [0.1, 0.7],
        [0.7, 0.1],
        [0.3, 0.9],
        [0.9, 0.3],
    ]
    c = np.array(c)
    obstacle_delta = 0.2
    obstacles_arrays_c.append(c)
    obstacle_delta_list.append(obstacle_delta)

    for ix, c in enumerate(obstacles_arrays_c):
        cost_fn, l = create_cost_function(c=c, obstacle_delta=obstacle_delta_list[ix])
        cost_fn_list.append(cost_fn)
        obstacle_l_list.append(l)

    return cost_fn_list, obstacle_l_list, obstacle_delta_list 


def create_cost_large_domains_all_conflicting(total_n_domains):
    assert total_n_domains == 3 # set up for 3 compelely conflicting domains
    cost_fn_list = []
    obstacle_l_list = []
    obstacle_delta_list = []
    obstacles_arrays_c = []

    # Obstacles Domain 1: 
    a = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    c = list(itertools.product(a, a))
    c.append([0.5, 0.5])
    c.append([0.5, 0.6])
    c.append([0.5, 0.4])
    c.append([0.6, 0.5])
    c.append([0.4, 0.5])
    c = np.array(c)  
    obstacle_delta = 0.13
    obstacles_arrays_c.append(c)
    obstacle_delta_list.append(obstacle_delta)

    # Obstacles Domain 2: 
    a = [0.1, 0.3, 0.5, 0.7, 0.9]
    c = list(itertools.product(a, a))
    c.append([0.5, 0.6])
    c.append([0.5, 0.4])
    c.append([0.6, 0.5])
    c.append([0.4, 0.5])
    c = np.array(c)  
    obstacle_delta = 0.13
    obstacles_arrays_c.append(c)
    obstacle_delta_list.append(obstacle_delta)

    # Obstacles Domain 3: 
    c = [
        [0.1, 0.7],
        [0.7, 0.1],
        [0.3, 0.7],
        [0.7, 0.3],
        [0.1, 0.9],
        [0.9, 0.1],
        [0.3, 0.9],
        [0.9, 0.3],
        [0.1, 0.5],
        [0.5, 0.1],
        [0.9, 0.5],
        [0.5, 0.9],
        [0.5, 0.3],
        [0.5, 0.7],
    ]
    c = np.array(c)
    obstacle_delta = 0.2
    obstacles_arrays_c.append(c)
    obstacle_delta_list.append(obstacle_delta)


    for ix, c in enumerate(obstacles_arrays_c):
        cost_fn, l = create_cost_function(c=c, obstacle_delta=obstacle_delta_list[ix])
        cost_fn_list.append(cost_fn)
        obstacle_l_list.append(l)

    return cost_fn_list, obstacle_l_list, obstacle_delta_list 


def create_multiple_domains_mocobo(
    n_domains, 
    n_points=30, 
    force_start=True, 
    force_goal=False,
):
    if n_domains == 3:
        cost_fn_list, obstacle_l_list, obstacle_delta_list = create_cost_large_domains_all_conflicting(total_n_domains=n_domains)
    elif n_domains == 4:
        cost_fn_list, obstacle_l_list, obstacle_delta_list = create_cost_large_four_domains()
    else: # 8,12
        cost_fn_list, obstacle_l_list, obstacle_delta_list = create_cost_large_load_saved_obstacle_arrays(total_n_domains=n_domains)

    start = np.zeros(2) + 0.05 # same for all 
    goal = np.array([0.95, 0.95]) # same for all 
    domains_list = []
    for i, cost_fn in enumerate(cost_fn_list):
        traj = PointBSpline(dim=2, num_points=n_points)
        domain = RoverDomain(
            cost_fn,
            start=start,
            goal=goal,
            traj=traj,
            force_start=force_start,
            force_goal=force_goal,
        )
        domain.obstacle_l = obstacle_l_list[i]
        domain.obstacle_delta = obstacle_delta_list[i]
        domains_list.append(domain)
    return domains_list


def create_cost_large_load_saved_obstacle_arrays(total_n_domains): 
    assert total_n_domains in [8,12], "saved obstacle arrays only provided for T=8 or T=12 obstacle course setups"
    obstacles_dir = f"../tasks/utils/rover_obstacle_arrays/t{total_n_domains}"
    assert os.path.exists(f"{obstacles_dir}/domain1.npy"), f"path to saved obstacle array {obstacles_dir}/domain1.npy does not exist"
    
    obstacles_arrays_c = []
    for i in range(total_n_domains):
        path_to_c = f"{obstacles_dir}/domain{i+1}.npy"
        c = np.load(path_to_c)
        obstacles_arrays_c.append(c)
    
    obstacle_delta_list = np.load(f"{obstacles_dir}/obstacle_deltas.npy")
    obstacle_delta_list = obstacle_delta_list.tolist()
    

    cost_fn_list = []
    obstacle_l_list = []
    for ix, c in enumerate(obstacles_arrays_c):
        cost_fn, l = create_cost_function(c=c, obstacle_delta=obstacle_delta_list[ix])
        cost_fn_list.append(cost_fn)
        obstacle_l_list.append(l)

    return cost_fn_list, obstacle_l_list, obstacle_delta_list 