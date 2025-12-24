import os
import pickle
import time
import numpy as np
import multiprocessing as mp

import env_TA as ce
from worker import run_episode_worker  # Import the worker function

def save_partial_results(save_path, completed_episodes, obj_vals, interdictions, times, states):
    """Save partial results to pickle file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    results = {
        "optimal_obj_vals": np.array(obj_vals[:completed_episodes], dtype=float),
        "all_optimal_interdiction_edges": list(interdictions[:completed_episodes]),
        "optimal_solution_times": list(times[:completed_episodes]),
        "last_episode": completed_episodes - 1,
        "states": states[:completed_episodes]
    }
    
    with open(save_path, "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    # User settings
    graphName = "G5x5"
    env_params = {'deterministic_agent': False,
                  'multiple_interdiction_attempts': False,
                  'attacker_strategy': 'zero_sum',  # canalize   isolate   divert  zero_sum
                  'training_budget_range': (5, 10),  #G5x5: zero_sum/isolate: (5,10), canalize/divert: (8,16) G10x10: zero_sum/isolate: (15,30), canalize/divert: (20,40)   #UKR: zero_sum/isolate: (10,20), canalize/divert: (15,25)
                  'max_path_length': 6,  #G5x5: 6,  G10x10: 13, UKR: 16
                 }
    
    # Run parameters
    num_of_scenarios = 100
    save_interval = 50
    max_workers = 4 #min(mp.cpu_count(), 8)
    
    # File paths
    current_dir = os.getcwd()
    save_filename = f"{graphName}_{env_params['attacker_strategy']}_solution.pkl"
    save_path = os.path.join(current_dir, '..', 'Solutions', save_filename)
    print(f"Will save to: {save_path}")
    
    # Load graph data
    node_filename = f"{graphName}_Nodes.csv"
    edge_filename = f"{graphName}_Edges.csv"
    nodes, edges = ce.create_nodes_edges(node_filename, edge_filename)
    
    # Prepare argument list for workers
    worker_args = [
        (episode, nodes, edges, env_params) 
        for episode in range(num_of_scenarios)
    ]
    
    # Initialize result storage
    optimal_obj_vals = [np.nan] * num_of_scenarios
    all_optimal_interdiction_edges = [tuple()] * num_of_scenarios
    optimal_solution_times = [np.nan] * num_of_scenarios
    all_states = [None] * num_of_scenarios  # Add this line to store states

    print(f"Starting {num_of_scenarios} episodes with {max_workers} workers...")
    
    # Use spawn context for better cluster compatibility
    ctx = mp.get_context("spawn")
    
    with ctx.Pool(processes=max_workers) as pool:
        # Submit all jobs and collect results as they complete
        result_objects = [pool.apply_async(run_episode_worker, (args,)) for args in worker_args]
        
        completed = 0
        for i, result_obj in enumerate(result_objects):
            try:
                episode_idx, obj_val, interdiction_edges, solve_time, state = result_obj.get()
                
                optimal_obj_vals[episode_idx] = obj_val
                all_optimal_interdiction_edges[episode_idx] = interdiction_edges
                optimal_solution_times[episode_idx] = solve_time
                all_states[episode_idx] = state  # Add this line to save the state

                completed += 1
                
                if completed % 25 == 0 or completed == num_of_scenarios:
                    print(f"Completed: {completed}/{num_of_scenarios} episodes", flush=True)
                
                if completed % save_interval == 0 or completed == num_of_scenarios:
                    save_partial_results(
                        save_path, completed, 
                        optimal_obj_vals, all_optimal_interdiction_edges, optimal_solution_times, all_states
                    )
                    print(f"Progress saved at episode {completed}", flush=True)
                    
            except Exception as e:
                print(f"Episode {i} failed with error: {e}", flush=True)
                completed += 1
    
    print(f"All {num_of_scenarios} episodes completed!")