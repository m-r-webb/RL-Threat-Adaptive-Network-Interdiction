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
        num_cpus=45,
        _temp_dir=os.environ.get('RAY_TMPDIR'), # Add this parameter with a short path
    )

# Import custom environment .py file
import env_TA as ce #modified for curriculum learning

#User Determined Settings
graphName = "UKR"
env_params = {'deterministic_agent': False,
              'multiple_interdiction_attempts': False,
              'attacker_strategy': 'isolate',  # canalize   isolate   divert  zero_sum
              'training_budget_range': (15, 30),  #G5x5: zero_sum/isolate: (5,10), canalize/divert: (8,16) G10x10: zero_sum/isolate: (15,30), canalize/divert: (20,40)   #UKR: zero_sum/isolate: (10,20), canalize/divert: (15,25)
              'max_path_length': 16,  #G5x5: 6,  G10x10: 13, UKR: 16
             }

# Number of scenarios to generate
num_of_scenarios = 500 
save_interval = 1  # Save every 10 episodes
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
save_filename = f"{graphName}_{env_params['attacker_strategy']}_solution_v12_19_1.pkl"
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
        #optimal_obj_val, optimal_interdiction_edges = env.solve_optimal_interdiction()
        optimal_obj_val, optimal_interdiction_edges = env.solve_backward_induction_ray(verbose=False, n_workers = 35)
        end_time = time.perf_counter()
    else:
        start_time = time.perf_counter()
        optimal_obj_val, optimal_interdiction_edges = env.solve_backward_induction_ray(verbose=False, n_workers = 35)
        end_time = time.perf_counter()
    
    solve_time = end_time - start_time

    #Save optimal solution value and interdiction set
    optimal_obj_vals[episode] = optimal_obj_val
    optimal_solution_times[episode] = solve_time
    all_optimal_interdiction_edges[episode] = frozenset(optimal_interdiction_edges)
    all_states[episode] = obs  # Add this line to save the state

    # Periodically save results
    if (episode + 1) % save_interval == 0 or (episode + 1) == num_of_scenarios:
        save_partial_results(save_path, episode+1,
                             optimal_obj_vals, all_optimal_interdiction_edges, 
                             optimal_solution_times, all_states)
        print(f"Progress saved at episode {episode+1} to {save_path}", flush=True)