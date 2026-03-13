# Import all required packages
import pandas as pd                   # For data manipulation and analysis
import io
import gymnasium as gym

#from gymnasium import spaces
import numpy as np
import os
import pickle
import copy
import random
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.ticker as ticker
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from tqdm import tqdm
from scipy.stats import ttest_ind
import seaborn as sns

from stable_baselines3.common.vec_env import DummyVecEnv

import ray

#User Determined Settings
graphName = "G5x5"
env_params = {'deterministic_agent': False,
              'multiple_interdiction_attempts': False,
              'attacker_strategy': 'zero_sum',  # canalize   isolate   divert  zero_sum
              'training_budget_range': (5,15),  #G5x5: zero_sum/isolate: (5,15), canalize/divert: (12,24) G10x10: zero_sum/isolate: (10,20), canalize/divert: (20,40)   #UKR: zero_sum/isolate: (10,20), canalize/divert: (18,30) G15x15: zero_sum/isolate: (25,45)
              'max_path_length': 2,  #G5x5: 2,  G10x10: 3, UKR: 4
              'sample_size': None,
              'penalty_value': -0.01,
             }
              
version_date = "02_19" # numeric month_day
version_type = "opt_m"  # bi for backward induction or opt_m or opt_d for optimal MIP
num_cpus_run = 36   # 36  44

model_name = "G5x5_S_MaskablePPO_zero_sum_B_v01_16_GCN"

# Number of scenarios to generate
num_of_scenarios = 500 
save_interval = 1  # Save every 10 episodes

# Print settings for easy copying for continuation runs
print("\n# --- USER DETERMINED SETTINGS (FOR CONTINUATION) ---")
print(f'graphName = "{graphName}"')
print(f"env_params = {env_params}")
print(f'version_date = "{version_date}"')
print(f'version_type = "{version_type}"')
print(f"num_cpus_run = {num_cpus_run}")
print(f"model_name = {model_name}")
print("# --------------------------------------------------\n")

if not ray.is_initialized():
    ray.init(
        address=None, 
        ignore_reinit_error=True, 
        logging_level='INFO',
        include_dashboard=False,
        num_cpus=num_cpus_run,
        _temp_dir=os.environ.get('RAY_TMPDIR'), # Add this parameter with a short path
    )

# Import custom environment .py file
import env_TA as ce #modified for curriculum learning

if version_type == "opt_m":
    opt_method = 'monolithic'  
elif version_type == "opt_d":
    opt_method = 'decomposition'

# LOAD Previous saved model
current_dir = os.getcwd()
models_dir = os.path.join(current_dir, '..', 'Trained_RL_Models')
model_timesteps = None #17640000 #None #4830000 #None #          # Set a number, 1512000, or  None
if model_timesteps == None:
    model_path = f"{models_dir}/{model_name}/best_model"
else:    model_path = f"{models_dir}/{model_name}/{model_name}_{model_timesteps}_steps"

if env_params['multiple_interdiction_attempts'] == True:
    MI_letter = 'M'
else:
    MI_letter = 'B'
    
current_dir = os.getcwd()

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

# Construct the path to the data files
save_filename = f"{graphName}_{env_params['attacker_strategy']}_{MI_letter}_solution_v{version_date}_{version_type}.pkl"
save_path = os.path.join(current_dir, '..', 'Solutions', save_filename)
print(f"Target solution file: {save_filename}")

# Create nodes and edges
node_filename = f"{graphName}_Nodes.csv"  # Dynamically include graphName
edge_filename = f"{graphName}_Edges.csv"  # Dynamically include graphName

nodes, edges = ce.create_nodes_edges(node_filename, edge_filename)

# Load Environment
env = ce.CustomEnv(nodes, edges, **env_params)

# -----------------------------------------------------------------------------
# CONTINUATION LOGIC
# -----------------------------------------------------------------------------
start_episode = 0
optimal_obj_vals = [np.nan] * num_of_scenarios
all_optimal_interdiction_edges = [tuple()] * num_of_scenarios
optimal_solution_times = [np.nan] * num_of_scenarios
all_states = [None] * num_of_scenarios

if os.path.exists(save_path):
    print(f"Loading existing results from {save_path}")
    try:
        with open(save_path, "rb") as f:
            existing_results = pickle.load(f)
        
        # Check if the file contains what we expect
        if "optimal_obj_vals" in existing_results:
             loaded_vals = list(existing_results["optimal_obj_vals"])
             loaded_count = len(loaded_vals)
             
             # If we have more existing results than requested scenarios, we might just stop or truncate?
             # Assuming we want to generate UP TO num_of_scenarios.
             
             if loaded_count >= num_of_scenarios:
                 print(f"File already contains {loaded_count} scenarios (>= requested {num_of_scenarios}).")
                 start_episode = num_of_scenarios # Loop won't run
                 # Load everything anyway just in case user increases num_of_scenarios later
                 optimal_obj_vals[:num_of_scenarios] = loaded_vals[:num_of_scenarios]
                 
                 if "all_optimal_interdiction_edges" in existing_results:
                     all_optimal_interdiction_edges[:num_of_scenarios] = list(existing_results["all_optimal_interdiction_edges"])[:num_of_scenarios]
                 if "optimal_solution_times" in existing_results:
                     optimal_solution_times[:num_of_scenarios] = list(existing_results["optimal_solution_times"])[:num_of_scenarios]
                 if "states" in existing_results:
                     all_states[:num_of_scenarios] = list(existing_results["states"])[:num_of_scenarios]
                     
             else:
                 print(f"Found {loaded_count} existing solutions. Continuing from episode {loaded_count}.")
                 start_episode = loaded_count
                 
                 # Fill the beginning of the lists with loaded data
                 optimal_obj_vals[:start_episode] = loaded_vals
                 
                 if "all_optimal_interdiction_edges" in existing_results:
                     all_optimal_interdiction_edges[:start_episode] = list(existing_results["all_optimal_interdiction_edges"])
                 if "optimal_solution_times" in existing_results:
                     optimal_solution_times[:start_episode] = list(existing_results["optimal_solution_times"])
                 if "states" in existing_results:
                     all_states[:start_episode] = list(existing_results["states"])

        else:
             print("Warning: 'optimal_obj_vals' not found in existing results. Starting from scratch.")

    except Exception as e:
        print(f"Error loading existing results: {e}. Starting from scratch.")
        start_episode = 0
else:
    print("No existing results found. Starting from episode 0.")

# -----------------------------------------------------------------------------
# MAIN GENERATION LOOP
# -----------------------------------------------------------------------------

if start_episode >= num_of_scenarios:
    print(f"All {num_of_scenarios} scenarios already completed. Exiting.")
else:
    try:
        for episode in range(start_episode, num_of_scenarios):
            try:
                print(f"Processing episode {episode+1}/{num_of_scenarios}", flush=True)
                
                obs = env.reset(seed=episode)
                env.render(indices = 3)
                
                time_limit_sec = 3600 # 1 hour time limit
                
                if env_params['attacker_strategy'] == "zero_sum":
                    start_time = time.perf_counter()
                    if version_type == 'opt_m':
                        optimal_obj_val, optimal_interdiction_edges = env.solve_optimal_interdiction(method=opt_method, time_limit=time_limit_sec) 
                    elif version_type == "opt_d":
                        optimal_obj_val, optimal_interdiction_edges = env.solve_optimal_interdiction(method=opt_method, time_limit=time_limit_sec) 
                    elif version_type == 'bi':
                        # Reduced n_workers to avoid thread resource exhaustion
                        optimal_obj_val, optimal_interdiction_edges = env.solve_backward_induction_ray(verbose=False, n_workers = num_cpus_run-6, enable_memoization=True, enable_outcome_caching=True, enable_alpha_pruning=True, jitter=True, rl_model_path=model_path, time_limit=time_limit_sec) 
                    end_time = time.perf_counter()
                else:
                    start_time = time.perf_counter()
                    # Reduced n_workers to avoid thread resource exhaustion
                    optimal_obj_val, optimal_interdiction_edges = env.solve_backward_induction_ray(verbose=False, n_workers = num_cpus_run-6, enable_memoization=True, enable_outcome_caching=True, enable_alpha_pruning=True, jitter=True, rl_model_path=model_path, time_limit=time_limit_sec) 
                    end_time = time.perf_counter()
                
                solve_time = end_time - start_time
            
                #Save optimal solution value and interdiction set
                optimal_obj_vals[episode] = optimal_obj_val
                optimal_solution_times[episode] = solve_time
                all_optimal_interdiction_edges[episode] = sorted(list(optimal_interdiction_edges))
                all_states[episode] = obs  # Add this line to save the state
            
                # Periodically save results
                if (episode + 1) % save_interval == 0 or (episode + 1) == num_of_scenarios:
                    save_partial_results(save_path, episode+1,
                                         optimal_obj_vals, all_optimal_interdiction_edges, 
                                         optimal_solution_times, all_states)
                    print(f"Progress saved at episode {episode+1} to {save_path}", flush=True)

            except Exception as e:
                print(f"Error in episode {episode}: {e}")
                # Save whatever we have
                save_partial_results(save_path, episode,
                                     optimal_obj_vals, all_optimal_interdiction_edges, 
                                     optimal_solution_times, all_states)
                raise e
    except Exception as e:
        print(f"Critical error outside loop: {e}")
        raise e
    finally:
        if ray.is_initialized():
            ray.shutdown()