# worker.py
import os
import time
import env_TA as ce

def run_episode_worker(args):
    """
    Worker function for running a single episode.
    """
    episode_idx, nodes, edges, env_params = args
    
    # Create fresh environment in this worker process
    env = ce.CustomEnv(
        nodes, edges, **env_params
    )
    
    # Reset with episode-specific seed
    env.reset(seed=episode_idx)
    
    # Solve and time the episode
    start_time = time.perf_counter()
    if env_params['attacker_strategy'] == "zero_sum":
        optimal_obj_val, optimal_interdiction_edges = env.solve_optimal_interdiction()
        #optimal_obj_val, optimal_interdiction_edges = env.solve_backward_induction(verbose=False)
    else:
        optimal_obj_val, optimal_interdiction_edges = env.solve_backward_induction(verbose=False)
    end_time = time.perf_counter()
    
    solve_time = end_time - start_time
    
    # Return canonical format
    return episode_idx, float(optimal_obj_val), tuple(sorted(frozenset(optimal_interdiction_edges))), solve_time, env.state