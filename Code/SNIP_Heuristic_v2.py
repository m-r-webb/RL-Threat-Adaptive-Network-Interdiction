# Import all required packages
import pandas as pd                   # For data manipulation and analysis
import gurobipy as grb                # Gurobi optimization library for solving mathematical models
import io
import gymnasium as gym

from gymnasium import spaces
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
import collections

from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_ind
import seaborn as sns

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from sb3_contrib import MaskablePPO

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import FlattenObservation
from sb3_contrib.common.maskable.utils import get_action_masks

# Import custom zero counting function
def count_zeros(arr):
    return np.count_nonzero(arr < 0.00001)


# Import custom environment .py file
#import custom_env as ce #original
#import custom_env_F as ce #modified for curriculum learning
import env_IM as ce #modified for curriculum learning

# Inputs
graphName = "G5x5"

# Environement Characteristics
env_deterministic = False
env_fixed_costs = False
env_curriculum_training = False
env_initial_budget = None
env_multiple_interdiction = False
env_max_budget = 25

# Get the current working directory
current_dir = os.getcwd()
models_dir = os.path.join(current_dir, '..', 'Trained_RL_Models')


# Graph nodes and edges to use
node_filename = f"{graphName}_Nodes.csv"  # Dynamically include graphName
edge_filename = f"{graphName}_Edges.csv"  # Dynamically include graphName

# Create nodes and edges
nodes, edges = ce.create_nodes_edges(node_filename, edge_filename)

env = ce.CustomEnv(nodes, edges, deterministic_agent=env_deterministic,
                   fixed_costs= env_fixed_costs,
                   curriculum_training=env_curriculum_training,
                   initial_budget=env_initial_budget,
                   max_training_budget=env_max_budget,                   
                   multiple_interdiction_attempts=env_multiple_interdiction)
eval_env = ce.CustomEnv(nodes, edges, deterministic_agent=env_deterministic,
                   fixed_costs= env_fixed_costs,
                   curriculum_training=env_curriculum_training,
                   initial_budget=env_initial_budget,
                   max_training_budget=env_max_budget,                   
                   multiple_interdiction_attempts=env_multiple_interdiction)

#Parallelized SNIP(IB) and Heuristic - add save for results, test with GPU core, then adapt for SNIP(IM)

# Worker function for parallel execution
def run_episode(episode, max_flow_val, optimal_obj_vals_ep):
    # Load Environment
    

    obs, _ = eval_env.reset(seed=episode)
    
    # Episode processing (same as original loop body)
    #max_flow_val = eval_env.reference_obj
    
    # Optimal solution calculation
    #start_optimal_time = time.perf_counter()
    #optimal_obj_vals_ep = np.zeros(30)
    #for i in range(30):
    #    optimal_obj_vals_ep[i], optimal_interdiction_edges = eval_env.solve_stochastic_max_flow(n_scenarios=200, seed=i*episode)

    #end_optimal_time = time.perf_counter()
    optimal_obj_var = np.var(optimal_obj_vals_ep, ddof=1)
    optimal_solution_time = 1#end_optimal_time - start_optimal_time
    
    # Greedy Heuristic (Eliminate Greatest Expected Capacity per Unit of Cost)
    action_list = []
    start_time = time.perf_counter()
    budget = obs['budget'][0]
    
    while budget >= 3:
        # Initialize a dictionary to accumulate sums for each key
        flow_sums = collections.defaultdict(float)

        # Run solve_max_flow 100 times and accumulate results
        for _ in range(num_runs):
            _, flow_vals = eval_env.solve_max_flow()
            for key, value in flow_vals.items():
                if key in eval_env.interdictable_edges:
                    flow_sums[key] += value

        # Compute mean values for each key
        flow_means = {key: flow_sums[key] / num_runs for key in flow_sums}

        # Zero out values for keys not in interdictable_edges
        #for key in flow_means:
         #   if key not in eval_env.interdictable_edges:
          #      flow_means[key] = 0.0

        # Find the key with the maximum mean value
        action_key = max(flow_means, key=flow_means.get)
        action = eval_env.interdictable_edges.index(action_key)

        obs, reward, done, _, _ = eval_env.step(action)
        budget = obs['budget'][0]
        

        action_list.append(action_key)
            
        if done:
            break
    
    end_time = time.perf_counter()
    agent_solution_time = end_time - start_time
    
    # Evaluation
    interim_objective_value = np.empty(1000)
    for i in range(1000):
        interim_objective_value[i], _ = eval_env.solve_max_flow()
    agent_best_reward = np.mean(interim_objective_value)

    # Add in worker function:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", 
            message="Precision loss occurred in moment calculation", 
            category=RuntimeWarning
        )
        t_stat, p_val = ttest_ind(optimal_obj_vals_ep, interim_objective_value, equal_var=False)
    
    return {
        'episode': episode,
        'max_flow_val': max_flow_val,
        'optimal_obj_vals': optimal_obj_vals_ep,
        'optimal_obj_var': optimal_obj_var,
        'optimal_solution_time': optimal_solution_time,
        #'optimal_edges': frozenset(optimal_interdiction_edges),
        'agent_solution_time': agent_solution_time,
        'agent_actions': frozenset(action_list),
        'agent_best_reward': agent_best_reward,
        'p_val': p_val,
        'actions_taken': action_list
    }

# Main parallel execution
if __name__ == '__main__':
    num_of_scenarios = 10000
    num_workers = multiprocessing.cpu_count()
    
    # Initialize results containers
    max_flow_vals = np.zeros(num_of_scenarios)
    optimal_obj_vals = np.zeros((num_of_scenarios, 30))
    optimal_obj_vars = np.zeros(num_of_scenarios)
    agent_best_rewards = np.zeros(num_of_scenarios)
    agent_solution_times = np.zeros(num_of_scenarios)
    optimal_solution_times = np.zeros(num_of_scenarios)
    all_optimal_edges = {}
    all_agent_actions = {}

    # Load arrays for stochastic
    max_flow_vals = np.load('max_flow_vals_S.npy')
    optimal_obj_vals = np.load('optimal_obj_vals_S.npy')
    
    action_tally = {i: 0 for i in env.interdictable_edges}

    optimal_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_episode, ep, max_flow_vals[episode], optimal_obj_vals[episode, :]) 
                  for ep in range(num_of_scenarios)]
        
        for future in tqdm(futures, total=num_of_scenarios, desc="Processing episodes"):
            result = future.result()
            ep = result['episode']
            
            # Store results
            max_flow_vals[ep] = result['max_flow_val']
            optimal_obj_vals[ep] = result['optimal_obj_vals']
            optimal_obj_vars[ep] = result['optimal_obj_var']
            optimal_solution_times[ep] = result['optimal_solution_time']
            #all_optimal_edges[ep] = result['optimal_edges']
            agent_solution_times[ep] = result['agent_solution_time']
            all_agent_actions[ep] = result['agent_actions']
            agent_best_rewards[ep] = result['agent_best_reward']
            
            # Update action tally
            for action in result['actions_taken']:
                action_tally[action] += 1
            
            if result['p_val'] >= 0.05:
                optimal_count += 1

    # Post-processing and output (same as original)
    relative_errors = abs(agent_best_rewards - np.mean(optimal_obj_vals, axis=1)) / (max_flow_vals - np.mean(optimal_obj_vals, axis=1))
    mean_relative_error = np.nanmean(relative_errors)
    
    print("Optimal Time to Solve (Mean):", np.mean(optimal_solution_times))
    print("\nHeuristic:", optimal_count/num_of_scenarios)
    print("Mean Relative Error:", mean_relative_error)
    print("RL Time to Solve (Mean):", np.mean(agent_solution_times))

    # Save arrays as .npy files
    #np.save('max_flow_vals_S.npy', max_flow_vals)
    #np.save('optimal_obj_vals_S.npy', optimal_obj_vals)
    #np.save('optimal_obj_vars_S.npy', optimal_obj_vars)
    np.save('agent_best_rewards_S_v2.npy', agent_best_rewards)

    # Save frozenset list as .pkl file
    #with open('all_optimal_interdiction_edges_S.pkl', 'wb') as f:
    #    pickle.dump(all_optimal_edges, f)
    with open('all_agent_actions_S_v2.pkl', 'wb') as f:
        pickle.dump(all_agent_actions, f)