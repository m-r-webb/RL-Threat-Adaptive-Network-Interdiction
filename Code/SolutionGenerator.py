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
if not ray.is_initialized():
    ray.init(
        address=None, 
        ignore_reinit_error=True, 
        logging_level='INFO',
        include_dashboard=False,
        num_cpus=46, #46  38
        _temp_dir=os.environ.get('RAY_TMPDIR'), # Add this parameter with a short path
    )

# Import custom environment .py file
import env_TA as ce #modified for curriculum learning

#User Determined Settings
graphName = "G15x15"
env_params = {'deterministic_agent': False,
              'multiple_interdiction_attempts': False,
              'attacker_strategy': 'zero_sum',  # canalize   isolate   divert  zero_sum
              'training_budget_range': (25,45),  #G5x5: zero_sum/isolate: (5,15), canalize/divert: (12,24) G10x10: zero_sum/isolate: (10,20), canalize/divert: (20,40)   #UKR: zero_sum/isolate: (10,20), canalize/divert: (18,30) G15x15: zero_sum/isolate: (25,45)
              'max_path_length': 2,  #G5x5: 2,  G10x10: 3, UKR: 4
              'sample_size': None,
              'penalty_value': -0.01,
             }
              
version_date = "02_10" # numeric month_day
version_type = "opt_d"  # bi for backward induction or opt_m or opt_d for optimal MIP
opt_method = 'decomposition'  # monolithic,  decomposition

# Number of scenarios to generate
num_of_scenarios = 500 
save_interval = 1  # Save every 10 episodes

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
print(save_filename)

# Create nodes and edges
node_filename = f"{graphName}_Nodes.csv"  # Dynamically include graphName
edge_filename = f"{graphName}_Edges.csv"  # Dynamically include graphName

nodes, edges = ce.create_nodes_edges(node_filename, edge_filename)

# Load Environment
env = ce.CustomEnv(nodes, edges, **env_params)

optimal_obj_vals = [np.nan] * num_of_scenarios
all_optimal_interdiction_edges = [tuple()] * num_of_scenarios
optimal_solution_times = [np.nan] * num_of_scenarios
all_states = [None] * num_of_scenarios  # Add this line to store states

for episode in range(num_of_scenarios):
    obs = env.reset(seed=episode)
    env.render(indices = 3)
    if env_params['attacker_strategy'] == "zero_sum":
        start_time = time.perf_counter()
        if version_type == 'opt_d' or version_type == 'opt_m':
            optimal_obj_val, optimal_interdiction_edges = env.solve_optimal_interdiction(method=opt_method)  
        elif version_type == 'bi':
            optimal_obj_val, optimal_interdiction_edges = env.solve_backward_induction_ray(verbose=False, n_workers = 38) #38,32)
        end_time = time.perf_counter()
    else:
        start_time = time.perf_counter()
        optimal_obj_val, optimal_interdiction_edges = env.solve_backward_induction_ray(verbose=False, n_workers = 38) #38,32)
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