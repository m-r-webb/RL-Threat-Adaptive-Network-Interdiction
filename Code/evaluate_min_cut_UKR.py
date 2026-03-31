import numpy as np
import os
import pickle
import copy
import time
from tqdm import tqdm

import env_TA as ce

def count_zeros(arr):
    return np.count_nonzero(arr < 0.00001)

def main():
    graphName = "UKR"
    env_params = {
        'deterministic_agent': False,
        'multiple_interdiction_attempts': False,
        'attacker_strategy': 'isolate',
        'training_budget_range': (10, 20),
        'max_path_length': 4,
        'sample_size': None,
        'penalty_value': -0.01,
    }
    
    solution_name = 'v03_20_bi'
    method_type = 'Min-Cut_Heuristic'
    
    print(f"Graph: {graphName}")
    print(f"Solution Name: {solution_name}")
    print(f"Attacker Strategy: {env_params['attacker_strategy']}")
    print(f"Method Type: {method_type}")
    
    # Create nodes and edges
    node_filename = f"{graphName}_Nodes.csv"
    edge_filename = f"{graphName}_Edges.csv"
    nodes, edges = ce.create_nodes_edges(node_filename, edge_filename)
    
    # Load Environment
    env = ce.CustomEnv(nodes, edges, **env_params)
    
    current_dir = os.getcwd()
    
    # Load Optimal Solutions for Comparison
    MI_letter = 'M' if env_params['multiple_interdiction_attempts'] else 'B'
    save_filename = f"{graphName}_{env_params['attacker_strategy']}_{MI_letter}_solution_{solution_name}.pkl"   
    save_path = os.path.join(current_dir, '..', 'Solutions', save_filename)
    
    print(f"Loading solution from: {save_path}")
    with open(save_path, "rb") as f:
        results = pickle.load(f)
        
    optimal_obj_vals = results["optimal_obj_vals"]
    all_optimal_interdiction_edges = results["all_optimal_interdiction_edges"]
    optimal_solution_times = results["optimal_solution_times"]
    all_states = results['states']
    
    num_of_scenarios = 500
    print(f"\nEvaluating Min-Cut Heuristic over {num_of_scenarios} episodes...")
    
    reference_objs = np.zeros(num_of_scenarios)
    agent_best_rewards = np.zeros(num_of_scenarios)
    agent_solution_times = np.zeros(num_of_scenarios)
    all_agent_actions = [None] * num_of_scenarios
    
    for episode in range(num_of_scenarios):
        # Print intermediate results every 10 episodes
        if episode > 0 and episode % 10 == 0:
            # We calculate current validation metric for the completed episodes up to 'episode'
            current_ref_objs = reference_objs[0:episode]
            current_opt_vals = optimal_obj_vals[0:episode]
            current_agent_rews = agent_best_rewards[0:episode]
            
            denom = current_ref_objs - current_opt_vals
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_errors = ((current_ref_objs - current_opt_vals) - (current_ref_objs - current_agent_rews)) / denom
                # Equivalent to np.clip, handle where denom == 0
                rel_errors = np.clip(rel_errors, 0, None)
            
            current_mean_error = np.nanmean(rel_errors)
            accuracy = count_zeros(rel_errors) / len(rel_errors)
            print(f"Episode {episode} | Current Mean Rel Error: {current_mean_error:.4f} | Current Accuracy: {accuracy:.4f}")
            
        env_state = copy.deepcopy(all_states[episode][0])
        obs, _ = env.load_network_from_state(episode, copy.deepcopy(env_state))
        
        reference_objs[episode] = env.reference_obj
        
        if optimal_obj_vals[episode] == reference_objs[episode]:
            agent_best_rewards[episode] = np.nan
            agent_solution_times[episode] = np.nan
            all_agent_actions[episode] = []
            continue
            
        start_time = time.perf_counter()
        objVal, actions_taken = env.solve_min_cut_heuristic()
        end_time = time.perf_counter()
        
        agent_solution_times[episode] = (end_time - start_time)
        all_agent_actions[episode] = actions_taken
        
        if all_agent_actions[episode] == all_optimal_interdiction_edges[episode]:
            objVal = optimal_obj_vals[episode]
        else:
            objVal, _ = env._calculate_isolate_objective_and_flows()
            
        agent_best_rewards[episode] = objVal

    # Final summary over all episodes
    print(f"\n--- Final Results over {num_of_scenarios} Episodes ---")
    denom = reference_objs[0:num_of_scenarios] - optimal_obj_vals[0:num_of_scenarios]
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_errors = ((reference_objs[0:num_of_scenarios] - optimal_obj_vals[0:num_of_scenarios]) - 
                           (reference_objs[0:num_of_scenarios] - agent_best_rewards[0:num_of_scenarios])) / denom
        relative_errors = np.clip(relative_errors, 0, None)
    
    mean_relative_error = np.nanmean(relative_errors)
    mean_sol_time = np.nanmean(agent_solution_times)
    final_accuracy = count_zeros(relative_errors) / num_of_scenarios
    
    print(f"Method: {method_type} on {graphName}")
    print(f"Mean Solution Time: {mean_sol_time:.4f} seconds")
    print(f"Mean Relative Error: {mean_relative_error:.4f}")
    print(f"Accuracy: {final_accuracy:.4f}")

if __name__ == '__main__':
    main()
