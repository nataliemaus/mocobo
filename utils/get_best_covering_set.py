import torch
from itertools import combinations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_SANITY_CHECKS = False 

def get_coverage_score(k_ys):
    k_ys = k_ys.to(device)
    # m_ys.shape == (k, T) 
    #   k = size of covering set
    #   T = n_objectives
    # output coverage score: best/max val acheived for each of the T objectives among the 4, sum all T of those. 
    return k_ys.max(0).values.sum().item()


def brute_force_get_best_covering_set_correct(y, k): 
    """
    Brute force algorithm for selecting k rows to maximize the sum of column-wise maxima.
    O(N choose k)
    
    Parameters:
        y (torch.Tensor): A (N, T) tensor of floats.
        k (int): Number of rows to select.
    
    Returns:
        best_selected_indices (list): Indices of the best selected rows.
        best_value (float): Objective value of the best solution (best coverage score)
    """
    y = y.to(device)
    all_combos = torch.tensor(list(combinations(range(y.shape[0]), k))) # torch.Size([210, 4]) (n_combos, k)
    try:
        all_combos = all_combos.to(device)
        all_options = y[all_combos] # torch.Size([210, 4, 11]) (n_combos, k, T)
        coverages = all_options.max(-2).values.sum(-1) # (210,) (n_combos,) score for each combo 
        best_coverage_ix = coverages.argmax()
        best_value = coverages[best_coverage_ix].item()
        selected_indices = all_combos[best_coverage_ix].tolist()
    except: # for specific case with so many combinations that we get GPU oom (i.e. N=512))
        all_options = y.cpu()[all_combos.cpu()]
        coverages = all_options.max(-2).values.sum(-1) # (210,) (n_combos,) score for each combo 
        best_coverage_ix = coverages.argmax()
        best_value = coverages[best_coverage_ix].item()
        selected_indices = all_combos[best_coverage_ix].tolist()
    

    if RUN_SANITY_CHECKS:
        best_value_check = get_coverage_score(y[selected_indices])
        assert best_value == best_value_check 

    return selected_indices, best_value


def update_best_covering_set(train_y, k, current_best_indicies=None, current_best_value=None, max_n_brute_fore=100):
    """
    Parameters:
        y (tensor): Input matrix of shape (N, T).
        k (int): Number of rows to select.
        current_best_indicies (list of ints): list of the k best indicies of the best covering set found so far during opt run 
        current_best_value (float): best converage score observed so far (cover score of the data points at current_best_indicies)
    
    Returns:
        best_selected_indices (list): Indices of the best selected rows.
        best_value (float): Objective value of the best solution (best coverage score)
    """
    
    N = train_y.shape[0]
    # 1. if N is small enough, just use the correct brute-force method rather than approxmating 
    if N <= max_n_brute_fore: 
        return brute_force_get_best_covering_set_correct(y=train_y, k=k)

    # 2. Use correct brute force comp for most recently collected data + current covering set of points 
    most_recent_data_indices = torch.arange(N) # indivies of ys to pass into correct bf algo
    num_recent_data_points = max_n_brute_fore - k
    most_recent_data_indices = most_recent_data_indices[-num_recent_data_points:].tolist() 
    if current_best_indicies is not None: 
        # Also include current set of k best indicies seen so far --> end up doing brute force comp on at most max_n_brute_fore points 
        most_recent_data_indices = most_recent_data_indices + current_best_indicies
        most_recent_data_indices = list(set(most_recent_data_indices)) # remove duplicates 
    bf_y = train_y[most_recent_data_indices]
    bf_selected_indices_not_converted, bf_best_val = brute_force_get_best_covering_set_correct(y=bf_y, k=k)
    # convert selected indicies to correct indicies for train_y 
    bf_selected_indicies = [most_recent_data_indices[i] for i in bf_selected_indices_not_converted]
    if RUN_SANITY_CHECKS:
        assert bf_best_val == get_coverage_score(train_y[bf_selected_indicies])
        if current_best_value is not None:
            # bf_best_val can't be lower bc we passed in current best ys as options into correct bf comp 
            assert bf_best_val >= current_best_value 

    # 3. Use approximate algorithm 1 on all data 
    A_star = algo_1_greedy_approxmation(y=train_y, k=k,)
    approx_selected_indicies = list(A_star)
    approx_best_val = get_coverage_score(train_y[approx_selected_indicies])

    # 4. return max/best between brute force on recent and approx Algo 1 on all
    if approx_best_val > bf_best_val:
        return approx_selected_indicies, approx_best_val
    return bf_selected_indicies, bf_best_val


def algo_1_greedy_approxmation(y, k):
    """
    Parallelized Greedy (1 - 1/e)-Approximation Algorithm for Finding S^*_{D_s}
    
    Parameters:
        y (tensor): Input matrix of shape (N, T) 
        k (int): Covering set size
    
    Returns:
        A (set[int]): Indices of selected data points
    """
    y = y.to(device)
    
    N, T = y.shape
    A = set()  # Store selected indices
    current_max = torch.full((T,), y.min(), device=y.device)  # Initial coverage (T,)

    # Mask for tracking unselected points
    mask = torch.ones(N, dtype=torch.bool, device=y.device)  # (N) True if unselected

    for _ in range(k):
        # Compute marginal gains in parallel
        new_max = torch.maximum(current_max.unsqueeze(0), y)  # (N, T)
        gains = new_max.sum(dim=1) - current_max.sum()  # (N,)
        # Mask out already selected points 
        gains[~mask] = -torch.inf  
        # Greedily select the x with biggest gain 
        x_best = torch.argmax(gains).item()
        # add point with best gain to approx covering set A 
        A.add(x_best)
        # Update max 
        current_max = new_max[x_best]
        # Mark selected element
        mask[x_best] = False 

    return A


def algo_1_greedy_approxmation_slower_unparallelized_version_not_used(y, k):
    """
    Greedy (1 - 1/e)-Approximation Algorithm for Finding S^*_{D_s} (Incremental Strategy)
    (Algorithm 1 in the MOCOBO paper)
    
    Parameters:
        y (tensor): Input matrix of shape (N, T) givien the T observed objective values for
            each of the N datapoints in the dataset D_s collected so far, 
        k (int): Covering set size 
    
    Returns:
        A (set[int]): the approxmate best covering set A^*_{D_s} 
            specifically a set of the K integers that are the indicies 
            of the K selected data points in D_s 
    """
    y = y.to(device)
    N, T = y.shape
    # 1. initialize approximate best covering set A 
    A = set() 
    # 2. initial coverage set to min value observed y.min()
    #  (don't actually use torch.zeros bc rewards y can be negative)
    #   NOTE: coverage score = current_max.sum()
    current_max = torch.ones(T).to(y.device) * y.min() # 
    # 3. loop k times to add k solutions to A 
    for _ in range(k):
        # 4. initialize x_best to None and best_gain to -inf
        x_best = None # index of best point to add to A 
        best_gain = -torch.inf
        # 5. loop over all N points, except those already in A 
        for j in range(N):
            if j in A:
                continue
            # 6. Compute marginal coverage of adding x to A (incremental gain)
            new_max = torch.maximum(current_max, y[j])
            gain = new_max.sum() - current_max.sum()
            # 7. if incremental gain is better than best we have seen
            if gain > best_gain:
                # 8. update best_gain and best_x 
                best_gain = gain
                x_best = j
            # 9. end if 
        # 10. end for 
        # 11. Add x_best to A, Update current max 
        A.add(x_best)
        current_max = torch.maximum(current_max, y[x_best]) 
    # 12. end for 
    # 13. return final approximate best covering set A 
    return A 
