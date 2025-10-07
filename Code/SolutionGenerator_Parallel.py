import os
import pickle
import time
import numpy as np
import multiprocessing as mp

import env_TA as ce
from worker import run_episode_worker  # Import the worker function

def save_partial_results(save_path, completed_episodes, obj_vals, interdictions, times):
    """Save partial results to pickle file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    results = {
        "optimal_obj_vals": np.array(obj_vals[:completed_episodes], dtype=float),
        "all_optimal_interdiction_edges": list(interdictions[:completed_episodes]),
        "optimal_solution_times": list(times[:completed_episodes]),
        "last_episode": completed_episodes - 1
    }
    
    with open(save_path, "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    # User settings
    graphName = "G4x5"
    env_deterministic = False
    env_initial_budget = None
    env_multiple_interdiction = False
    env_attacker_strategy = 'divert'  # canalize, isolate, divert, zero_sum  
    
    # Run parameters
    num_of_scenarios = 1000
    save_interval = 10
    max_workers = 24 #min(mp.cpu_count(), 8)
    
    # File paths
    current_dir = os.getcwd()
    save_filename = f"{graphName}_{env_attacker_strategy}_solution.pkl"
    save_path = os.path.join(current_dir, '..', 'Solutions', save_filename)
    print(f"Will save to: {save_path}")
    
    # Load graph data
    node_filename = f"{graphName}_Nodes.csv"
    edge_filename = f"{graphName}_Edges.csv"
    nodes, edges = ce.create_nodes_edges(node_filename, edge_filename)
    
    # Pack environment parameters
    env_params = (env_deterministic, env_multiple_interdiction, env_attacker_strategy, env_initial_budget)
    
    # Prepare argument list for workers
    worker_args = [
        (episode, nodes, edges, env_params) 
        for episode in range(num_of_scenarios)
    ]
    
    # Initialize result storage
    optimal_obj_vals = [np.nan] * num_of_scenarios
    all_optimal_interdiction_edges = [tuple()] * num_of_scenarios
    optimal_solution_times = [np.nan] * num_of_scenarios
    
    print(f"Starting {num_of_scenarios} episodes with {max_workers} workers...")
    
    # Use spawn context for better cluster compatibility
    ctx = mp.get_context("spawn")
    
    with ctx.Pool(processes=max_workers) as pool:
        # Submit all jobs and collect results as they complete
        result_objects = [pool.apply_async(run_episode_worker, (args,)) for args in worker_args]
        
        completed = 0
        for i, result_obj in enumerate(result_objects):
            try:
                episode_idx, obj_val, interdiction_edges, solve_time = result_obj.get()
                
                optimal_obj_vals[episode_idx] = obj_val
                all_optimal_interdiction_edges[episode_idx] = interdiction_edges
                optimal_solution_times[episode_idx] = solve_time
                
                completed += 1
                
                if completed % 5 == 0 or completed == num_of_scenarios:
                    print(f"Completed: {completed}/{num_of_scenarios} episodes", flush=True)
                
                if completed % save_interval == 0 or completed == num_of_scenarios:
                    save_partial_results(
                        save_path, completed, 
                        optimal_obj_vals, all_optimal_interdiction_edges, optimal_solution_times
                    )
                    print(f"Progress saved at episode {completed}", flush=True)
                    
            except Exception as e:
                print(f"Episode {i} failed with error: {e}", flush=True)
                completed += 1
    
    print(f"All {num_of_scenarios} episodes completed!")