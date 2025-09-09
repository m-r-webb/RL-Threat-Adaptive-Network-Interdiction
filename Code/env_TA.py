#Purpose: Gymnasium environment for single/multiple attempt max flow deterministic/stochastic network interdiction by zero-sum/threat adaptive attackers

# Import all required packages
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress most logs (including CUDA errors)

import pandas as pd
import gurobipy as grb                # Gurobi optimization library for solving mathematical models
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
import random
import networkx as nx

import tensorflow as tf
tf.get_logger().setLevel('ERROR')          # Optional: Suppress Python-

import torch as th
import time
from collections import defaultdict


# Class representing Node Object
class Node():
    def __init__(self, ID, xpos, ypos, node_type):
        self.ID = ID                  # Node's ID
        self.xpos = xpos               # Node's x position
        self.ypos = ypos               # Node's y position
        self.node_type = node_type    # Node's type

# Class representing Edge Object
class Edge():
    def __init__(self, ID, interdictable, capacity=300, interdicted=0, interdiction_cost=100, interdiction_probability=0):
        self.ID = ID                  # Edge's ID
        self.interdictable = interdictable
        self.capacity = capacity      # Edge's capacity
        self.interdicted = interdicted          # Edge is not interdicted by default
        self.interdiction_cost = interdiction_cost    # Edge's resources cost to interdict
        self.interdiction_probability = interdiction_probability # Edge's susceptibility to interdiction

def create_nodes_edges(node_filename, edge_filename):

    #import os
    
    # Get the current working directory
    current_dir = os.getcwd()

    # Construct the path to the data files
    node_data_path = os.path.join(current_dir, '..', 'Network_Data', node_filename)
    edge_data_path = os.path.join(current_dir, '..', 'Network_Data', edge_filename)

    # Read the CSV file
    nodes_df = pd.read_csv(node_data_path)
    edges_df = pd.read_csv(edge_data_path)

    nodes = dict()
    for i, row in nodes_df.iterrows():
        nodes[row['node']] = Node(ID = row['node'],   # Node's ID
        xpos = row['x_pos'],                      # Node's x position
        ypos = row['y_pos'],                      # Node's y position
        node_type = row['type'],                  # Node's type
        )

    edges = dict()
    for i, row in edges_df.iterrows():
        edge_id = (int(row['Origin']), int(row['Destination']))  # Create tuple for edge ID
        edges[edge_id] = Edge(
            ID = edge_id,                              # Edge's ID as tuple
            interdictable = row['Interdictable'],      # Edge's susceptibility to interdiction
        )

    return nodes, edges

#TO DO: - Update solve optimal flow() for undirected edges

#Create a custom gymnasium environment for the RL agent
class CustomEnv(gym.Env):
    def __init__(self, nodes, edges, deterministic_agent=True, fixed_costs=True, curriculum_training=False, initial_budget = None, 
                 max_training_budget = 50, min_training_budget = 3, multiple_interdiction_attempts=True, attacker_strategy="zero_sum"):
        super(CustomEnv, self).__init__()
        self.nodes = nodes
        self.edges_reset = edges
        self.edges_episode = copy.deepcopy(self.edges_reset)
        self.multiple_interdiction_attempts = multiple_interdiction_attempts
        self.attacker_strategy = attacker_strategy
        
        self.source_nodes = [1]  #[]
        self.sink_nodes = [len(self.nodes)] #[]
        self.intermediate_nodes = list(range(2,len(self.nodes)))            
        
        # Determine list of edges susceptible to interdiction
        self.all_edges = list({(u, v) for u, v in self.edges_reset.keys()}.union(
                      {(v, u) for u, v in self.edges_reset.keys() if u not in self.source_nodes and v not in self.sink_nodes}
))
        
        self.interdictable_edges = []
        self.edge_departures =[]
        self.edge_arrivals = []
        
        for key, edge in self.edges_reset.items():
            if edge.interdictable == 1:
                self.interdictable_edges.append(key)
                self.edge_departures.append(key[0])
                self.edge_arrivals.append(key[1])
        
        self.source_edges = [e for e in self.all_edges if e[0] in self.source_nodes]
        self.sink_edges = [e for e in self.all_edges if e[1] in self.sink_nodes]

        out_edges = defaultdict(list)
        in_edges = defaultdict(list)
        for edge in self.all_edges:
            out_edges[edge[0]].append(edge)
            in_edges[edge[1]].append(edge)

        self.edge_groups ={node_id: {
            'out': out_edges.get(node_id, []),
            'in': in_edges.get(node_id, [])}
                           for node_id in self.nodes}
        
        # Determine if interdiction outcomes are deterministic or stochastic
        self.deterministic_outcomes = deterministic_agent
        # Determine if edge costs are fixed or variable
        self.fixed_costs = fixed_costs
        
        # Observation space dimensions:
        self.num_interdictable_edges = len(self.interdictable_edges)
        
        # Updated Observation space to allow model to scale betweeen network sizes
        self.max_num_edges = self.num_interdictable_edges 
        self.max_num_nodes = self.sink_nodes[0]
        
        self.edge_capacity_space = spaces.Box(low=0, high=100, shape=(self.max_num_edges,), dtype=int)  # u vector with capacity

        if self.multiple_interdiction_attempts:
            self.edge_interdicted_space = spaces.Box(low=0, high=10, shape=(self.max_num_edges,), dtype=int)  # v vector (binary interdiction state)
        else:
            self.edge_interdicted_space = spaces.MultiBinary(self.max_num_edges)  # v vector (binary interdiction state)
        
        self.edge_costs_space = spaces.Box(low=3, high=5, shape=(self.max_num_edges,), dtype=int)  # arc interdiction costs
        self.edge_interdiction_probability_space = spaces.Box(low=0, high=1, shape=(self.max_num_edges,), dtype=float)  # arc interdiction costs
        
        self.budget_space = spaces.Box(low=0, high=100, shape=(1,), dtype=int)  # Remaining resources budget

        self.edge_departure_node_space = spaces.Box(low=1, high=self.max_num_nodes, shape=(self.max_num_edges,), dtype=int)  #high=len(self.nodes)
        self.edge_arrival_node_space = spaces.Box(low=1, high=self.max_num_nodes, shape=(self.max_num_edges,), dtype=int) #high=len(self.nodes)

        
        # Precompute edge-to-index mapping once
        self.edge_to_index = {edge: idx for idx, edge in enumerate(self.interdictable_edges)}

        # Combine all into a Dict space
        self.observation_space = spaces.Dict({
            'edge_capacity': self.edge_capacity_space,
            'edge_interdicted': self.edge_interdicted_space,
            'edge_costs': self.edge_costs_space,
            'edge_interdiction_probability': self.edge_interdiction_probability_space,
            'edge_departure_node': self.edge_departure_node_space,
            'edge_arrival_node': self.edge_arrival_node_space,
            'budget': self.budget_space
        })

        self.e = grb.Env(params={"OutputFlag": 0, "LogToConsole": 0, "Threads":2})

        # Actions are just to interdict or do nothing
        if self.attacker_strategy == "zero_sum":
            self.action_space = spaces.Discrete(self.max_num_edges)
        else:
            self.action_space = spaces.Discrete(self.max_num_edges + 1) #add do nothing action

        self.rng = np.random.default_rng()
        self.num_stochastic_scenarios = None
        self.num_stochastic_scenarios_IM = None

        self.max_training_budget = max_training_budget
        self.min_training_budget = min_training_budget

        # Add curriculum learning parameters
        self.curriculum_training = curriculum_training
        if self.curriculum_training == True:
            self.zero_prob = 0.5  # Initial probability of setting edge capacity to 0
            self.zero_prob_decay = 0.95  # Decay rate per episode
            self.min_zero_prob = 0.1  # Minimum probability
            self.initial_budget = 3
        else:
            self.zero_prob = 0  # Initial probability of setting edge capacity to 0
            self.zero_prob_decay = 1  # Decay rate per episode
            self.min_zero_prob = 0  # Minimum probability
            self.initial_budget = initial_budget

    # Maskable Actions - Not Updated for Threat Adaptive
    def action_masks(self) -> np.ndarray:
        mask = []
        for edge in self.interdictable_edges:
            edge_obj = self.edges_episode[edge]
            if self.multiple_interdiction_attempts:
                mask = self.state['edge_capacity'] * self.state['edge_interdiction_probability'] * np.where(self.state['edge_costs'] > self.state['budget'][0],0,1)
                mask = np.where(mask==0,0,1)
            else:
                mask = (1-self.state['edge_interdicted']) * self.state['edge_capacity'] * self.state['edge_interdiction_probability'] * np.where(self.state['edge_costs'] > self.state['budget'][0],0,1)
                mask = np.where(mask==0,0,1)
            return np.array(mask, dtype=np.bool_)
    
    # Begin Curriculum Learning Methods
    def set_zero_prob(self, prob):
        """External method to update zero probability"""
        self.zero_prob = max(self.min_zero_prob, min(1.0, prob))

    def decay_zero_prob(self):
        """Decay the zero probability"""
        self.zero_prob = max(self.min_zero_prob, self.zero_prob * self.zero_prob_decay)

    def increase_budget(self, multiplier=1.2):
        self.initial_budget = min(8, int(self.initial_budget * multiplier))
    # End Curriculum Learning Methods

    
    def solve_max_flow(self):  
        """Solve the Max Flow network problem, output objective value and edge flows"""
        if not hasattr(self, 'maxflow_model'):
            # Build model and store variables/constraints
            self.maxflow_model = grb.Model("Max Flow", env=self.e)

            self.mf_all_edges = self.all_edges
            self.mf_all_edges.append((len(self.nodes),1))
            
            self.flow_var = self.maxflow_model.addVars(
                self.mf_all_edges,
                vtype=grb.GRB.CONTINUOUS, lb=0, name="flow_var")
                
            self.maxflow_model.addConstrs(
                (grb.quicksum(self.flow_var[e] for e in self.edge_groups[n]['out']) == grb.quicksum(self.flow_var[e] for e in self.edge_groups[n]['in'])
                 for n in self.intermediate_nodes),
                name="flow_conservation"
            )

            self.maxflow_model.addConstr(self.flow_var[(len(self.nodes),1)]-grb.quicksum(self.flow_var[e] for e in self.edge_groups[1]['out'])==0)
            self.maxflow_model.addConstr(-self.flow_var[(len(self.nodes),1)]+grb.quicksum(self.flow_var[e] for e in self.edge_groups[self.max_num_nodes]['in'])==0)
        
            self.maxflow_model.setObjective(self.flow_var[(len(self.nodes),1)], grb.GRB.MAXIMIZE) #Maximize Flow

        # Update Capacity Bounds  #IM CHANGE
        upper_bounds = np.random.binomial(1,
                                          (1 - self.state["edge_interdiction_probability"][:self.num_interdictable_edges]) ** 
                                          self.state["edge_interdicted"][:self.num_interdictable_edges]) * self.state["edge_capacity"][:self.num_interdictable_edges]

        if not hasattr(self, 'mf_capacity_constraints'):
            self.mf_capacity_constraints = {}
            for idx, e in enumerate(self.interdictable_edges):
                con = self.maxflow_model.addConstr(
                    self.flow_var[e] + self.flow_var[(e[1], e[0])] <= upper_bounds[idx],
                    name=f"flow_capacity_{e[0]}_{e[1]}"
                )
                self.mf_capacity_constraints[e] = con
        else:
            for idx, e in enumerate(self.interdictable_edges):
                self.mf_capacity_constraints[e].rhs = upper_bounds[idx]

        self.maxflow_model.update()
#        self.maxflow_model.write("maxflow_model.lp")

        # Optimize
        self.maxflow_model.optimize()

        # Convert Result into Flows
        flow_results = {e: var.X for e, var in self.flow_var.items()}
        
        return self.maxflow_model.ObjVal, flow_results

    def solve_optimal_interdiction(self):
        if self.deterministic_outcomes == True: #Solve Deterministic Case with Wood's Max/Min Formulation
            if not hasattr(self, 'optimal_deterministic_model'):
                # Initialize the Gurobi model
                self.optimal_deterministic_model = grb.Model("Network Interdiction Model 1D", env=self.e)
                
                # Define Decision Variables
                self.alpha = self.optimal_deterministic_model.addVars(self.nodes.keys(), vtype=grb.GRB.BINARY, name="alpha")
                self.beta = self.optimal_deterministic_model.addVars(self.edges_reset.keys(), vtype=grb.GRB.BINARY, name="beta")
                self.gamma = self.optimal_deterministic_model.addVars(self.interdictable_edges, vtype=grb.GRB.BINARY, name="gamma")
                
                # Define Constraints
                self.optimal_deterministic_model.addConstrs(
                    (self.alpha[e[0]] - self.alpha[e[1]] + self.beta[e] + 
                     (self.gamma[e] if e in self.interdictable_edges else 0) >= 0 for e in self.edges_reset.keys()),
                    name="flow_conservation"
                )

                self.optimal_deterministic_model.addConstrs(
                    (self.alpha[e[1]] - self.alpha[e[0]] + self.beta[e] + 
                     (self.gamma[e] if e in self.interdictable_edges else 0) >= 0 for e in self.edges_reset.keys()),
                    name="flow_conservation_reverse"
                )

                self.optimal_deterministic_model.addConstr(self.alpha[self.sink_nodes[0]]-self.alpha[self.source_nodes[0]] >=1,
                                                          name = "sink-source")
            
            # Update Constraints
            if hasattr(self, 'budget_constr'):
                self.optimal_deterministic_model.remove(self.budget_constr)

            self.budget_constr = self.optimal_deterministic_model.addConstr(
                grb.quicksum(self.edges_episode[e].interdiction_cost * self.gamma[e]
                             for e in self.interdictable_edges) <= self.remaining_budget[0],
                name="budget"
            )

            # Define Objective Value
            self.optimal_deterministic_model.setObjective(grb.quicksum(edge.capacity * self.beta[edge_id] for edge_id, edge in self.edges_episode.items()), grb.GRB.MINIMIZE)

            # Optimize
            self.optimal_deterministic_model.optimize()

            interdicted_edges = [
                e for e in self.interdictable_edges 
                if self.gamma[e].X > 0.99  # Account for floating point precision
            ]

            return self.optimal_deterministic_model.ObjVal, interdicted_edges
        
        else:  #Solve Stochastic Case with Cormican's Formulation          #PICKUP HERE!!!!
            M = 100                       # Number of training episodes
            N = 700                   # Number of test episodes
            seed_list = [100, 200, 300]#, 400, 500]
            best_objective_value = 100000
            best_interdicted_edges = []
            unique_interdicted_sets = []

            # Test multiple solutions
            for seed in seed_list:
                if self.multiple_interdiction_attempts:
                    objective_value, interdicted_edges, interdicted_quantities = self.solve_stochastic_max_flow_IM(n_scenarios=M, seed=seed)
                else:
                    objective_value, interdicted_edges = self.solve_stochastic_max_flow(n_scenarios=M, seed=seed)
                #print(seed,": ", objective_value, ", ", interdicted_edges)
                # Convert interdicted_edges to a frozenset for hashability
                interdicted_set = frozenset(interdicted_edges)
                    
                # Check if the set of interdicted edges is unique
                if interdicted_set not in unique_interdicted_sets:
                    unique_interdicted_sets.append(interdicted_set)       

                    if self.multiple_interdiction_attempts:
                        objective_value, interdicted_edges, interdicted_quantities = self.solve_stochastic_max_flow_IM(n_scenarios=N, interdicted_edges=interdicted_edges, interdicted_quantities=interdicted_quantities)
                    else:
                        objective_value, interdicted_edges = self.solve_stochastic_max_flow(n_scenarios=N, interdicted_edges=interdicted_edges)

                    if objective_value < best_objective_value:
                        best_objective_value = objective_value
                        best_interdicted_edges = interdicted_edges

            return best_objective_value, best_interdicted_edges

    def solve_stochastic_max_flow(self, n_scenarios = 50, seed = 173, interdicted_edges = []):
        # Optimally Solve for Stochastic Solution using Model 1D and SAA
        if not hasattr(self, 'optimal_stochastic_model'):
            # Initializing the model
            self.optimal_stochastic_model = grb.Model("Stochastic Model", env=self.e)

            # Creating decision variables
            self.stochastic_gamma = self.optimal_stochastic_model.addVars(self.interdictable_edges, vtype=grb.GRB.BINARY, name="gamma")

            # Create Variable Lower Bounds
            self.optimal_stochastic_model.setAttr("LB", [self.stochastic_gamma[e] for e in interdicted_edges],1)
            
             # Budget constraint
            self.stochastic_budget_constr = self.optimal_stochastic_model.addConstr(grb.quicksum(
                self.edges_episode[e].interdiction_cost * self.stochastic_gamma[e] 
                for e in self.interdictable_edges) <= self.remaining_budget[0], name="budget")

            self.stochastic_old_state = self.state
            self.stochastic_old_interdicted_edges = interdicted_edges

        if self.stochastic_old_interdicted_edges != interdicted_edges:
            # Update Variable Lower Bounds
            self.optimal_stochastic_model.setAttr("LB", [self.stochastic_gamma[e] for e in self.interdictable_edges],0)
            self.optimal_stochastic_model.setAttr("LB", [self.stochastic_gamma[e] for e in interdicted_edges],1)
            self.stochastic_old_interdicted_edges=interdicted_edges
        
        if self.num_stochastic_scenarios != n_scenarios:
            # Generate scenarios
            self.num_stochastic_scenarios = n_scenarios
            self.scenarios = range(n_scenarios)

            if hasattr(self, 'stochastic_alpha'):
                self.optimal_stochastic_model.remove(self.stochastic_alpha)
                self.optimal_stochastic_model.remove(self.stochastic_beta)
                self.optimal_stochastic_model.update()  # Force model synchronization
                del self.stochastic_alpha, self.stochastic_beta 
                
            self.stochastic_alpha = self.optimal_stochastic_model.addVars([(i, s) for s in self.scenarios for i in self.nodes], 
                                                  vtype=grb.GRB.BINARY, name="alpha")
            self.stochastic_beta = self.optimal_stochastic_model.addVars([(e, s) for s in self.scenarios for e in self.edges_reset],
                                                                          vtype=grb.GRB.BINARY, name="beta")

            if hasattr(self, 'stochastic_source_sink_constr'):
                self.optimal_stochastic_model.remove(self.stochastic_source_sink_constr)
                del self.stochastic_source_sink_constr 

            self.stochastic_source_sink_constr = self.optimal_stochastic_model.addConstrs((self.stochastic_alpha[self.sink_nodes[0],s] - self.stochastic_alpha[self.source_nodes[0], s] >= 1 for s in self.scenarios), name="source_sink")

            # Objective Function
            self.optimal_stochastic_model.setObjective((1/n_scenarios)*grb.quicksum(self.edges_episode[e].capacity * self.stochastic_beta[e, s]
                for s in self.scenarios
                for e in self.edges_reset), grb.GRB.MINIMIZE)
        
        # Scenario generation
        scenario_outcomes = np.random.binomial(1, self.state["edge_interdiction_probability"], 
                                               size=(n_scenarios, len(self.interdictable_edges))) #Generate a 1 for success and a 0 for failure
        
        if hasattr(self, 'stochastic_aabg_constr'):
            self.optimal_stochastic_model.remove(self.stochastic_aabg_constr)
            self.optimal_stochastic_model.remove(self.stochastic_aabg_reverse_constr)
            self.optimal_stochastic_model.update()  # Force model synchronization
            del self.stochastic_aabg_constr, self.stochastic_aabg_reverse_constr
            
        self.stochastic_aabg_constr = self.optimal_stochastic_model.addConstrs((self.stochastic_alpha[e[0],s] - self.stochastic_alpha[e[1], s]+self.stochastic_beta[e, s]+ (self.stochastic_gamma[e] * scenario_outcomes[s, self.edge_to_index[e]] if e in self.edge_to_index else 0)>=0 for s in self.scenarios for e in self.edges_reset.keys()), name='aabg')
        self.stochastic_aabg_reverse_constr = self.optimal_stochastic_model.addConstrs((self.stochastic_alpha[e[1],s] - self.stochastic_alpha[e[0], s]+self.stochastic_beta[e, s]+ (self.stochastic_gamma[e] * scenario_outcomes[s, self.edge_to_index[e]] if e in self.edge_to_index else 0)>=0 for s in self.scenarios for e in self.edges_reset.keys()), name='aabg')


        # Solving
        self.optimal_stochastic_model.optimize()

        interdicted_edges = [
            e for e in self.interdictable_edges
            if self.stochastic_gamma[e].X > 0.5  # Tolerate minor numerical issues
        ]

        return(self.optimal_stochastic_model.objVal, interdicted_edges)


    def solve_stochastic_max_flow_IM(self, n_scenarios = 50, seed = 173, interdicted_edges = [], interdicted_quantities =[]):
        # Optimally Solve for Stochastic Solution using Model 1D and SAA
        if not hasattr(self, 'optimal_stochastic_model_IM'):
            # Initializing the model
            self.optimal_stochastic_model_IM = grb.Model("Stochastic Model_IM", env=self.e)

            # Creating decision variables
            # Create composite keys: (edge_tuple, k)
            gamma_indices = [(e, k) for e in self.interdictable_edges for k in range(1, 9)]
            self.stochastic_gamma_IM = self.optimal_stochastic_model_IM.addVars(gamma_indices, vtype=grb.GRB.BINARY, name="g_IM")
            self.optimal_stochastic_model_IM.update()

            # Create Variable Lower Bounds
            self.optimal_stochastic_model_IM.setAttr("LB", [self.stochastic_gamma_IM[e,k] for e, k in zip(interdicted_edges, interdicted_quantities)],1)

            # Gamma constraint
            self.stochastic_gamma_constr_IM = self.optimal_stochastic_model_IM.addConstrs((grb.quicksum(
                self.stochastic_gamma_IM[e,k] for k in range(1,9)) <= 1 for e in self.interdictable_edges), name="gamma_constr_IM")
            
             # Budget constraint
            self.stochastic_budget_constr_IM = self.optimal_stochastic_model_IM.addConstr(grb.quicksum(
                self.edges_episode[e].interdiction_cost * k * self.stochastic_gamma_IM[e,k] 
                for e in self.interdictable_edges for k in range(1,9)) <= self.remaining_budget[0], name="budget_IM")

            self.stochastic_old_state_IM = self.state
            self.stochastic_old_interdicted_edges_IM = interdicted_edges
            self.stochastic_old_interdicted_quantities_IM = interdicted_quantities

        if self.stochastic_old_interdicted_edges_IM != interdicted_edges or self.stochastic_old_interdicted_quantities_IM != interdicted_quantities:
            # Update Variable Lower Bounds
            self.optimal_stochastic_model_IM.setAttr("LB", [self.stochastic_gamma_IM[e,k] for e in self.interdictable_edges for k in range(1,9)],0)
            self.optimal_stochastic_model_IM.setAttr("LB", [self.stochastic_gamma_IM[e,k] for e, k in zip(interdicted_edges, interdicted_quantities)],1)
            self.stochastic_old_interdicted_edges_IM=interdicted_edges
        
        if self.num_stochastic_scenarios_IM != n_scenarios:
            # Generate scenarios
            self.num_stochastic_scenarios_IM = n_scenarios
            self.scenarios_IM = range(n_scenarios)

            if hasattr(self, 'stochastic_alpha_IM'):
                self.optimal_stochastic_model_IM.remove(self.stochastic_alpha_IM)
                self.optimal_stochastic_model_IM.remove(self.stochastic_beta_IM)
                self.optimal_stochastic_model_IM.update()  # Force model synchronization
                del self.stochastic_alpha_IM, self.stochastic_beta_IM 
                
            self.stochastic_alpha_IM = self.optimal_stochastic_model_IM.addVars([(i, s) for s in self.scenarios_IM for i in self.nodes], 
                                                  vtype=grb.GRB.BINARY, name="alpha_IM")
            self.stochastic_beta_IM = self.optimal_stochastic_model_IM.addVars([(e, s) for s in self.scenarios_IM for e in self.edges_reset],
                                                                          vtype=grb.GRB.BINARY, name="beta_IM")

            if hasattr(self, 'stochastic_source_sink_constr_IM'):
                self.optimal_stochastic_model_IM.remove(self.stochastic_source_sink_constr_IM)
                del self.stochastic_source_sink_constr_IM

            self.stochastic_source_sink_constr_IM = self.optimal_stochastic_model_IM.addConstrs((self.stochastic_alpha_IM[self.sink_nodes[0],s] - self.stochastic_alpha_IM[self.source_nodes[0], s] >= 1 for s in self.scenarios_IM), name="source_sink_IM")

            # Objective Function
            self.optimal_stochastic_model_IM.setObjective((1/n_scenarios)*grb.quicksum(self.edges_episode[e].capacity * self.stochastic_beta_IM[e, s]
                for s in self.scenarios_IM
                for e in self.edges_reset), grb.GRB.MINIMIZE)
        
        # Scenario generation 
        # Compute base probability and ensure it's an array
        p_base = np.full(len(self.interdictable_edges), self.state["edge_interdiction_probability"])

        # Create k values (1 to 8)
        k_vals = np.arange(1, 9)

        # Calculate success probabilities: 1 - (1-p)^k for each edge and k
        probs = 1 - (1 - p_base[:, np.newaxis]) ** k_vals

        # Generate scenario outcomes
        scenario_outcomes = np.random.binomial(1, probs, size=(n_scenarios, len(self.interdictable_edges), len(k_vals)))
        
        if hasattr(self, 'stochastic_aabg_constr_IM'):
            self.optimal_stochastic_model_IM.remove(self.stochastic_aabg_constr_IM)
            self.optimal_stochastic_model_IM.remove(self.stochastic_aabg_reverse_constr_IM)

            self.optimal_stochastic_model_IM.update()  # Force model synchronization
            del self.stochastic_aabg_constr_IM, self.stochastic_aabg_reverse_constr_IM
            
        self.stochastic_aabg_constr_IM = self.optimal_stochastic_model_IM.addConstrs((self.stochastic_alpha_IM[e[0],s] - self.stochastic_alpha_IM[e[1], s]+self.stochastic_beta_IM[e, s]+ (grb.quicksum(self.stochastic_gamma_IM[e,k] * scenario_outcomes[s, self.edge_to_index[e],k-1] for k in k_vals) if e in self.edge_to_index else 0) >= 0 for s in self.scenarios_IM for e in self.edges_reset.keys()), name='aabg_IM')
        self.stochastic_aabg_reverse_constr_IM = self.optimal_stochastic_model_IM.addConstrs((self.stochastic_alpha_IM[e[1],s] - self.stochastic_alpha_IM[e[0], s]+self.stochastic_beta_IM[e, s]+ (grb.quicksum(self.stochastic_gamma_IM[e,k] * scenario_outcomes[s, self.edge_to_index[e],k-1] for k in k_vals) if e in self.edge_to_index else 0) >= 0 for s in self.scenarios_IM for e in self.edges_reset.keys()), name='aabg_IM')

        # Solving
        self.optimal_stochastic_model_IM.optimize()

        # Extract interdiction decisions with k-values
        interdiction_decisions = []
        for e in self.interdictable_edges:
            for k in range(1, 9):
                if self.stochastic_gamma_IM[e, k].X > 0.5:
                    interdiction_decisions.append((e, k))
                    break  # Only one k per edge possible

        # Extract just the edge list if needed
        interdicted_edges = [e for e, k in interdiction_decisions]
        interdicted_quantities = [k for e, k in interdiction_decisions]

        return (self.optimal_stochastic_model_IM.objVal, interdicted_edges, interdicted_quantities)
    
    # BEGIN Gymnasium Environment Methods        
    def reset(self, seed=None, options=None):
        if hasattr(self, 'master_model'):
            try:
                self.master_model.dispose()  # Properly free Gurobi resources
            except Exception:
                pass  # In case dispose() is not available, continue
            del self.master_model
        if hasattr(self, 'sub_model'):
            try:
                self.sub_model.dispose()  # Properly free Gurobi resources
            except Exception:
                pass  # In case dispose() is not available, continue
            del self.sub_model
        if hasattr(self, 'benders_cuts'):
            del self.benders_cuts
        if hasattr(self, 'optimal_stochastic_model'):
            try:
                self.optimal_stochastic_model.dispose()  # Properly free Gurobi resources
            except Exception:
                pass  # In case dispose() is not available, continue
            del self.optimal_stochastic_model
            self.num_stochastic_scenarios = None
            del self.stochastic_alpha, self.stochastic_beta
            del self.stochastic_source_sink_constr, self.stochastic_aabg_constr
        if hasattr(self, 'optimal_stochastic_model_IM'):
            try:
                self.optimal_stochastic_model_IM.dispose()  # Properly free Gurobi resources
            except Exception:
                pass  # In case dispose() is not available, continue
            del self.optimal_stochastic_model_IM
            self.num_stochastic_scenarios_IM = None

            del self.stochastic_alpha_IM, self.stochastic_beta_IM
            del self.stochastic_source_sink_constr_IM, self.stochastic_aabg_constr_IM

        super().reset(seed=seed)
        # Set seeds
        if seed is not None:
            self.edge_capacity_space.seed(seed)
            self.edge_costs_space.seed(seed)
            self.edge_interdiction_probability_space.seed(seed)
            self.budget_space.seed(seed)

        # Sample a network instance
        indicator = False
        while not indicator:
            
            # Sample arc capacities until network has a positive max flow
            max_flow_objective_value = 0
            #while max_flow_objective_value == 0:
            edge_capacities = self.edge_capacity_space.sample() #Sample arc capacities
            edge_capacities[self.num_interdictable_edges:self.max_num_edges] = 0 #mask extra edges

            # Modify capacity sampling with curriculum
            if self.curriculum_training == True:
                # Apply zero probability mask
                mask = self.rng.random(size=edge_capacities.shape) < self.zero_prob
                edge_capacities[mask] = 0
            
            # Update edge capacities in the graph
            for edge_id, edge in enumerate(self.interdictable_edges):
                self.edges_episode[edge].capacity = edge_capacities[edge_id]
                
            edge_interdicted = np.zeros(self.max_num_edges) #Initialize with no targeted edges  #self.num_interdictable_edges

            if self.fixed_costs:
                edge_costs = tuple(4 for _ in range(self.max_num_edges)) #Initialize with same costs  # self.num_interdictable_edges

            else:
                edge_costs = self.edge_costs_space.sample() #Sample arc costs

            #Sample interdiction probabilities or set as deterministic outcomes
            if self.deterministic_outcomes:
                edge_interdiction_probabilities = np.ones(self.max_num_edges, dtype=np.float32)  
            else:
                edge_interdiction_probabilities  = self.edge_interdiction_probability_space.sample()
                sample_rounded = np.round(edge_interdiction_probabilities * 4)  # Scale to integers 0-4
                edge_interdiction_probabilities = (sample_rounded.astype(float) / 4)  # Convert back to 0.25 increments
            
            # Update edge attributes
            for edge_id, edge in enumerate(self.interdictable_edges):                
                self.edges_episode[edge].interdiction_cost = edge_costs[edge_id]
                self.edges_episode[edge].interdiction_probability = edge_interdiction_probabilities[edge_id]

            #Sample budget space until an interdictable edge exists with cost less than remaining budget
            if self.initial_budget != None:
                self.remaining_budget = np.array([self.initial_budget], dtype=int)
            else:
                self.remaining_budget = self.budget_space.sample()
                self.remaining_budget[0] = round(((self.max_training_budget-self.min_training_budget)*self.remaining_budget[0]/100)+self.min_training_budget)

            departure_nodes = np.full(self.max_num_edges, self.max_num_nodes)
            arrival_nodes = np.full(self.max_num_edges, self.max_num_nodes)
            departure_nodes[:len(np.array(self.edge_departures))]=np.array(self.edge_departures)
            arrival_nodes[:len(np.array(self.edge_arrivals))]=np.array(self.edge_arrivals)
            
            # 3. Threat Strategy Specific Checks TO BE IMPLEMENTED

            # Check that the ideal interdiction is better than the uninterdicted max flow
            if self.deterministic_outcomes:
            #    optimal_interdiction_objective_value, interdicted_edges= self.solve_optimal_interdiction()
            #    if max_flow_objective_value > optimal_interdiction_objective_value: #least_resources <= remaining_budget and objective_value > 0:
                indicator = True #If all good, change indicator to True
            else:
                indicator = True

        self.state = {
            'edge_capacity': edge_capacities,
            'edge_interdicted': edge_interdicted,
            'edge_costs': edge_costs,
            'edge_interdiction_probability': edge_interdiction_probabilities,
            'edge_departure_node': departure_nodes, #np.array(self.edge_departures),
            'edge_arrival_node': arrival_nodes, #np.array(self.edge_arrivals),
            'budget': self.remaining_budget
        }
        self.reference_obj, _ = self.solve_max_flow()
        self.last_obj = self.reference_obj 
        self.reference_budget = self.remaining_budget[0]
        
        return self.state, {}  # Return initial state and an empty info dict

    def step(self, action):  
        done = False
        do_nothing = False
        reward = 0
        penalty = 0
        objective_value = None

        self.remaining_budget = self.state['budget']
                   
        # Determine if action is valid (not interdicted or too expensive)
        if self.multiple_interdiction_attempts: #IM CHANGE
            if self.state['edge_interdicted'][action] ==10 or self.remaining_budget[0] - self.state['edge_costs'][action]<-0.1 or self.state['edge_capacity'][action]==0 or action>=self.num_interdictable_edges: 
                valid_action = False
            else:
                valid_action = True
        else:
            if self.state['edge_interdicted'][action] ==1 or self.remaining_budget[0] - self.state['edge_costs'][action]<-0.1 or self.state['edge_capacity'][action]==0 or action>=self.num_interdictable_edges: 
                valid_action = False
            else:
                valid_action = True

        # Adjudicate the action's effect on the state space and the corresponding reward 
        if valid_action == True and do_nothing == False: # If Valid Edge Removal
            #Deduct edge cost from budget
            self.remaining_budget = np.array([self.remaining_budget[0] - self.state['edge_costs'][action]])
            #Annotate edge is targeted
            self.state['edge_interdicted'][action]+=1 #IM CHANGE
        else: # Invalid Action Attempted
            penalty = -1#-10  #Must be greater than the max capacity * the cardinality of the min cut of arcs
            self.remaining_budget[0] = max(0, self.remaining_budget[0] - 1)

        # Compute the minimum resources needed to remove another valid edge
        if self.multiple_interdiction_attempts: #IM CHANGE
            least_resources = 3
        else:
            masked_costs = self.state['edge_costs']*(1-self.state['edge_interdicted'])
            least_resources = min(masked_costs[masked_costs>0])
        
        ## Determine reward and if episode is complete
        if self.deterministic_outcomes and valid_action:
            objective_value, _ = self.solve_max_flow()
            reward = (self.last_obj - objective_value)/self.reference_budget #Modified reward function for EX20
            self.last_obj = objective_value
        elif not self.deterministic_outcomes and valid_action:
            if self.multiple_interdiction_attempts:
                edges_interdicted = (self.state['edge_interdicted'] > 0).astype(int)
                arr =((1-self.state['edge_interdiction_probability'])**self.state['edge_interdicted'])
                arr = edges_interdicted*arr
            else:
                arr =(self.state['edge_interdicted']*self.state['edge_interdiction_probability'])
            non_zero_elements = arr[arr!=0]
            if non_zero_elements.size >0:
                x = np.mean(non_zero_elements)
                if x <= 0.5:
                    # Linear interpolation from (0,1) to (0.5,20000)
                    iterations = int(1 + (1000 - 1) * (x / 0.5))
                else:
                    # Linear interpolation from (0.5,20000) to (1,1)
                    iterations =int(1000 - (1000 - 1) * ((x - 0.5) / 0.5))
            else:
                iterations = 1
            interim_objective_value = np.empty(iterations)
            for i in range(iterations):
                interim_objective_value[i], _ = self.solve_max_flow()
            objective_value = np.mean(interim_objective_value)
            reward = max(self.last_obj - objective_value, 0)/self.reference_budget #Modified reward function for EX20
            if reward > 0:
                self.last_obj = objective_value
        
        # Remaining Budget is insufficient for future valid action or Network is Deterministic and Max Flow is Zero
        if (self.remaining_budget[0] < least_resources or (self.deterministic_outcomes and objective_value == 0)):
            done = True
                
        # Update state observation
        self.state['budget']= self.remaining_budget
        
        reward = reward + penalty #Compute reward
        
        return self.state, float(reward), bool(done), False, {}  # state, reward, terminated, truncated, info

    def render(self, mode='human'):
        print(f"State: {self.state}")
    # END Gymnasium Environment Methods