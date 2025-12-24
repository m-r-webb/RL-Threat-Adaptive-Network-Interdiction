
import os
import numpy as np
import pandas as pd
from env_TA import CustomEnv, create_nodes_edges, Node, Edge

# Mock data creation if files are missing or just use existing ones
# The workspace has ukraine_railway_nodes.csv and ukraine_railway_edges.csv
# But create_nodes_edges expects them in '../Network_Data/' relative to CWD?
# Let's check create_nodes_edges implementation in env_TA.py

# def create_nodes_edges(node_filename, edge_filename):
#     # Get the current working directory
#     current_dir = os.getcwd()
#     # Construct the path to the data files
#     node_data_path = os.path.join(current_dir, '..', 'Network_Data', node_filename)
#     edge_data_path = os.path.join(current_dir, '..', 'Network_Data', edge_filename)

# This path construction might be problematic if I run from the Code directory.
# I'll just manually load the data or create dummy data.

def create_dummy_data():
    nodes = {
        1: Node(1, 0, 0, 'source'),
        2: Node(2, 1, 0, 'intermediate'),
        3: Node(3, 2, 0, 'sink')
    }
    edges = {
        (1, 2): Edge((1, 2), interdictable=1, capacity=10),
        (2, 3): Edge((2, 3), interdictable=1, capacity=10)
    }
    return nodes, edges

def test_memory_reduction():
    nodes, edges = create_dummy_data()
    
    # Test with reduce_memory_usage=True (Default)
    print("Testing with reduce_memory_usage=True...")
    env = CustomEnv(nodes, edges, reduce_memory_usage=True, max_num_nodes=3, max_num_edges=2)
    env.reset()
    
    # Force some interdiction probability so we get stochastic outcomes
    env.state['edge_interdiction_probability'][:] = 0.5
    env.state['edge_interdicted'][:] = 1 # Mark as interdicted so probability applies
    
    # Run calculation
    obj, flows = env._calculate_stochastic_objective_and_flow()
    
    # Check local cache
    print("Checking local cache...")
    for outcome, res in env.local_outcome_cache.items():
        if 'nonzero_flow_indices' in res:
            print(f"Outcome {outcome}: Found 'nonzero_flow_indices'. Success.")
        else:
            print(f"Outcome {outcome}: 'nonzero_flow_indices' NOT found. Failure.")
            
        if 'flows' in res:
             print(f"Outcome {outcome}: Found 'flows' (unexpected for memory reduction).")
        
    # Test with reduce_memory_usage=False
    print("\nTesting with reduce_memory_usage=False...")
    env_full = CustomEnv(nodes, edges, reduce_memory_usage=False, max_num_nodes=3, max_num_edges=2)
    env_full.reset()
    env_full.state['edge_interdiction_probability'][:] = 0.5
    env_full.state['edge_interdicted'][:] = 1
    
    obj_full, flows_full = env_full._calculate_stochastic_objective_and_flow()
    
    # Check local cache
    print("Checking local cache...")
    for outcome, res in env_full.local_outcome_cache.items():
        if 'flows' in res:
            print(f"Outcome {outcome}: Found 'flows'. Success.")
        else:
            print(f"Outcome {outcome}: 'flows' NOT found. Failure.")
            
        if 'nonzero_flow_indices' in res:
             print(f"Outcome {outcome}: Found 'nonzero_flow_indices' (unexpected for full mode).")

if __name__ == "__main__":
    test_memory_reduction()
