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
#import networkx as nx
from tqdm import tqdm
import tensorflow as tf
tf.get_logger().setLevel('ERROR')          # Optional: Suppress Python-

#import torch as th
#import time
from collections import defaultdict
from collections import Counter


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

#Create a custom gymnasium environment for the RL agent
class CustomEnv(gym.Env):
    """Custom Gym environment for network interdiction problems."""
    # Class constants
    DEFAULT_BUDGET_RANGE = (0, 100)
    DEFAULT_EDGE_CAPACITY_RANGE = (0, 100)
    DEFAULT_EDGE_COST_RANGE = (0, 10)
    DEFAULT_TRAINING_BUDGET_RANGE = (5, 10)
    DEFAULT_TRAINING_EDGE_CAPACITY_RANGE = (30, 60)
    DEFAULT_TRAINING_EDGE_COST_RANGE = (3, 5)
    MAX_INTERDICTION_ATTEMPTS = 10
    MAX_SOURCE_FLOW = 125
    MAX_SINK_NEED = 40
    GUROBI_ENV = grb.Env(params={"OutputFlag": 0, "LogToConsole": 0, "Threads": 1, "Seed": 1})
    PENALTY_VALUE = -0.1
    SAMPLE_SIZE = 1000
    max_num_edges = 500  # Maximum edges across all test graphs  Should work through G15x15
    max_num_nodes = 250    # Maximum nodes across all test graphs
    
    def __init__(self, nodes, edges, deterministic_agent=True, initial_budget = None, 
                 multiple_interdiction_attempts=True, attacker_strategy="zero_sum"):
        super(CustomEnv, self).__init__()

        #Setup core environment attributes
        self.nodes = nodes
        self.edges_reset = edges
        self.edges_episode = copy.deepcopy(self.edges_reset)        
        self.multiple_interdiction_attempts = multiple_interdiction_attempts
        self.attacker_strategy = attacker_strategy
        self.deterministic_outcomes = deterministic_agent
        self.initial_budget = initial_budget
        
        self.num_stochastic_scenarios = None
        self.num_stochastic_scenarios_IM = None
        self.old_routing = "none"
        
        # Initialize network structure
        self._setup_network_structure()

        # Setup observation and action spaces
        self._setup_spaces()

    def _cache_flow_array(self):
        """Fully vectorized cache using array indexing."""
        num_edges = self.num_interdictable_edges
    
        # Pre-allocate
        flow_array = np.zeros(num_edges, dtype=np.float32)
    
        # Vectorized extraction using list comprehension (compiled to C internally)
        edges = self.interdictable_edges
        flows = self.reference_flows
    
        # Batch get operations
        forward_keys = edges
        reverse_keys = [(e[1], e[0]) for e in edges]
    
        # Vectorized lookup and sum
        flow_array = np.array([flows.get(fk, 0) + flows.get(rk, 0) for fk, rk in zip(forward_keys, reverse_keys)], dtype=np.float32)
    
        self.cached_flow_array = flow_array
    
    def _setup_network_structure(self):
        """Initialize network nodes and edges structure."""
        # Define node types
        self.source_nodes = [1]
        self.sink_nodes = [len(self.nodes)]
        self.intermediate_nodes = list(range(2, len(self.nodes)))
        
        # Create all possible edges
        self.all_edges = list({(u, v) for u, v in self.edges_reset.keys()}.union(
                      {(v, u) for u, v in self.edges_reset.keys() if u not in self.source_nodes and v not in self.sink_nodes}))
        
        # Extract interdictable edges and their attributes
        self.interdictable_edges = []
        self.edge_departures =[]
        self.edge_arrivals = []
        
        for key, edge in self.edges_reset.items():
            if edge.interdictable == 1:
                self.interdictable_edges.append(key)
                self.edge_departures.append(key[0])
                self.edge_arrivals.append(key[1])

        self.all_interdictable_edges = list(self.interdictable_edges) + [(v, u) for (u, v) in self.interdictable_edges]

        
        self.source_edges = [e for e in self.all_edges if e[0] in self.source_nodes]
        self.sink_edges = [e for e in self.all_edges if e[1] in self.sink_nodes]
        
        # Create edge groups for efficient lookup
        out_edges = defaultdict(list)
        in_edges = defaultdict(list)
        for edge in self.all_edges:
            out_edges[edge[0]].append(edge)
            in_edges[edge[1]].append(edge)

        self.edge_groups ={node_id: {
            'out': out_edges.get(node_id, []),
            'in': in_edges.get(node_id, [])}
                           for node_id in self.nodes}
        
        # Create edge-to-index mapping
        self.edge_to_index = {edge: idx for idx, edge in enumerate(self.interdictable_edges)}

        for edge_id in self.edge_groups[self.sink_nodes[0]]['in']:
            self.edges_episode[edge_id].capacity = self.MAX_SINK_NEED

        self.edges_with_sink_source = list(self.edges_reset.keys()) + [(self.sink_nodes[0],self.source_nodes[0])]

        # Identify sink-connecting interdictable edges
        self.cached_sink_edge_indices = [
            idx for idx, edge in enumerate(self.interdictable_edges)
            if edge[1] in {e[0] for e in self.edge_groups[self.sink_nodes[0]]['in']}
        ]
        self.num_cached_sink_edge_indices = len(self.cached_sink_edge_indices)
            
    def _setup_spaces(self):
        """Setup observation and action spaces based on environment configuration."""
        # Calculate space dimensions
        self.num_interdictable_edges = len(self.interdictable_edges)
    
        # Store actual size for this specific graph
        self.actual_num_edges = self.num_interdictable_edges
        
        # Create base spaces
        self.base_spaces = self._create_base_spaces()
        
        # Create strategy-specific observation space
        self.observation_space = self._create_observation_space(self.base_spaces)
        
        # Create action space based on attacker strategy
        if self.attacker_strategy == "zero_sum":
            self.action_space = spaces.Discrete(self.max_num_edges)
        else:
            self.action_space = spaces.Discrete(self.max_num_edges + 1)  # Add "do nothing" action

    def _create_base_spaces(self):
        """Create the base observation spaces used across all strategies."""
        return {
            'edge_capacity': spaces.Box(low=self.DEFAULT_EDGE_CAPACITY_RANGE[0], high=self.DEFAULT_EDGE_CAPACITY_RANGE[1], 
                shape=(self.max_num_edges,), dtype=int),
            'edge_interdicted': self._create_interdiction_space(),
            'edge_costs': spaces.Box(low=self.DEFAULT_EDGE_COST_RANGE[0], high=self.DEFAULT_EDGE_COST_RANGE[1],
                                     shape=(self.max_num_edges,), dtype=int),
            'edge_interdiction_probability': spaces.Box(low=0, high=1, shape=(self.max_num_edges,), dtype=float),
            'budget': spaces.Box(low=self.DEFAULT_BUDGET_RANGE[0], high=self.DEFAULT_BUDGET_RANGE[1], shape=(1,), dtype=int),
            'edge_departure_node': spaces.Box(low=1, high=self.max_num_nodes, shape=(self.max_num_edges,), dtype=int),
            'edge_arrival_node': spaces.Box(low=1, high=self.max_num_nodes, shape=(self.max_num_edges,), dtype=int),
            'padding_mask': spaces.Box(low=0, high=1, shape=(self.max_num_edges,), dtype=np.float32)
        }

    def _create_interdiction_space(self):
        """Create interdiction space based on multiple attempts setting."""
        if self.multiple_interdiction_attempts:
            return spaces.Box(low=0, high=self.MAX_INTERDICTION_ATTEMPTS, shape=(self.max_num_edges,), dtype=int)
        else:
            return spaces.MultiBinary(self.max_num_edges)

    def _create_observation_space(self, base_spaces):
        """Create observation space based on attacker strategy."""
        self.strategy_spaces = {
            "zero_sum": {},
            "canalize": {'canalize_objective': spaces.MultiBinary(self.max_num_edges)},
            "isolate": {'isolate_objective': spaces.MultiBinary(self.max_num_edges)},
            "divert": {'divert_from_objective': spaces.MultiBinary(self.max_num_edges),
                       'divert_to_objective': spaces.MultiBinary(self.max_num_edges)}
        }
        
        # Combine base spaces with strategy-specific spaces
        observation_dict = {**self.base_spaces, **self.strategy_spaces.get(self.attacker_strategy, {})}
        return spaces.Dict(observation_dict)
    
    def solve_max_flow(self, capacity_dict=None,
                       interdicted_edges=None,
                       routing_assumption = "gurobi_default"):  
        """
        Solve the Max Flow network problem, output objective value and edge flows.
    
        Parameters:
        -----------
        capacity_dict : dict, optional
            If provided, uses this capacity dictionary instead of current state.
            Useful for batch solving with different capacity configurations.
        interdicted_edges : list, optional
            List of interdicted edges (for compatibility, can be ignored if using capacity_dict)
        routing_assumption : str
            Routing optimization objective ('gurobi_default', 'canalize', etc.)
    
        Returns:
        --------
        tuple: (objective_value, flow_dict)
        """
        # Initialize model on first call
        if not hasattr(self, 'maxflow_model'):
            self._initialize_maxflow_model()

        # Update capacity constraints
        if capacity_dict is not None:
            # Use provided capacity dict (optimized path)
            self._update_capacity_constraints_from_dict(capacity_dict)
        else:
            # Use current state (legacy path)
            self._update_capacity_constraints()

        # Update objectives based on routing assumption
        if self.old_routing != routing_assumption:
            self._set_routing_objectives(routing_assumption)
            
        # Solve and return results
        self.maxflow_model.params.Seed = 1
        self.maxflow_model.optimize()
        flow_results = {e: round(var.X) for e, var in self.flow_var.items()}
        
        return round(self.maxflow_model.ObjVal), flow_results 

    def _initialize_maxflow_model(self):
        """Initialize the Gurobi max flow model with variables and constraints."""
        self.maxflow_model = grb.Model("Max Flow", env=self.GUROBI_ENV)
        self.super_edge = (len(self.nodes), 1)
        
        # Prepare edge list with super sink-source connection
        self.mf_all_edges = self.all_edges + [self.super_edge] 

        ##VARIABLES
        # Add Flow variables
        self.flow_var = self.maxflow_model.addVars(self.mf_all_edges, vtype=grb.GRB.CONTINUOUS, lb=0, name="flow_var")

        # Add Edge Usage variables
        self.edge_used = self.maxflow_model.addVars(self.mf_all_edges, vtype=grb.GRB.BINARY, name="edge_used")

        ##CONSTRAINTS
        # Flow conservation for intermediate nodes
        self.maxflow_model.addConstrs(
            (grb.quicksum(self.flow_var[e] for e in self.edge_groups[n]['out']) == 
             grb.quicksum(self.flow_var[e] for e in self.edge_groups[n]['in'])
             for n in self.intermediate_nodes), name="flow_conservation"
        )
    
        # Source and sink flow conservation
        self.maxflow_model.addConstr(self.flow_var[self.super_edge] - grb.quicksum(self.flow_var[e] for e in self.edge_groups[1]['out']) == 0,
                                     name="source_conservation"
        )
    
        self.maxflow_model.addConstr(grb.quicksum(self.flow_var[e] for e in self.edge_groups[self.sink_nodes[0]]['in']) -
                                     self.flow_var[self.super_edge] == 0, name="sink_conservation"
        )
    
        # One-way flow constraints
        self.maxflow_model.addConstrs((self.edge_used[(u, v)] + self.edge_used[(v, u)] <= 1 
                                       for u, v in self.edges_reset.keys()  if u not in self.source_nodes and v not in self.sink_nodes),
                                      name="one_way_flow"
        )
    
        # Minimum flow forward and reverse constraints
        self.maxflow_model.addConstrs((self.flow_var[e] >= self.edge_used[e] for e in self.all_interdictable_edges), name="min_flow_forward")
    
        # Source and sink capacity limits
        self.maxflow_model.addConstr(self.flow_var[self.super_edge] <= self.MAX_SOURCE_FLOW, name="max_source_flow")
    
        self.maxflow_model.addConstrs((self.flow_var[e] <= self.MAX_SINK_NEED for e in self.edge_groups[self.sink_nodes[0]]['in']),
                                      name="sink_node_max_capacities"
        )

    def _update_capacity_constraints(self):
        """LEGACY Update edge capacity constraints based on current interdiction state."""
        # Calculate current edge capacities considering interdiction
        # Ensure probabilities are float to avoid integer power errors
        probs = self.state["edge_interdiction_probability"][:self.num_interdictable_edges].astype(float)
        interdicted = self.state["edge_interdicted"][:self.num_interdictable_edges].astype(int)
    
        # Compute success probabilities
        success_probs = (1.0 - probs) ** interdicted
    
        # Generate binomial outcomes
        upper_bounds = np.random.binomial(1, success_probs) * self.state["edge_capacity"][:self.num_interdictable_edges]
    
        # Remove old capacity constraints if they exist
        if hasattr(self, 'forward_cons'):
            self.maxflow_model.remove(self.forward_cons)

        # Single batch addition for forward constraints
        self.forward_cons = self.maxflow_model.addConstrs((
            self.flow_var[e] <= upper_bounds[idx % self.num_interdictable_edges] * self.edge_used[e] 
            for idx, e in enumerate(self.all_interdictable_edges)), name="flow_capacity_forward")

    def _update_capacity_constraints_from_dict(self, capacity_dict):
        """
        Update capacity constraints using provided capacity dictionary.
        Optimized for batch processing.
    
        Parameters:
        -----------
        capacity_dict : dict
            Mapping from edge tuple to capacity value
        """
        if hasattr(self, 'forward_cons'):
            self.maxflow_model.remove(self.forward_cons)

        self.forward_cons = self.maxflow_model.addConstrs((
            self.flow_var[e] <= capacity_dict.get(e, 0) * self.edge_used[e] 
            for idx, e in enumerate(self.all_interdictable_edges)), name="flow_capacity_forward")
        
    def _set_routing_objectives(self, routing_assumption):
        """Set model objectives based on routing assumption."""
        # Clear existing objectives
        self.maxflow_model.NumObj = 0
        self.maxflow_model.update()
        
        if routing_assumption == "gurobi_default":
            self.maxflow_model.setObjective(self.flow_var[self.super_edge], grb.GRB.MAXIMIZE)
        elif routing_assumption == "consolidated":
            # Primary: Maximize flow
            self.maxflow_model.setObjectiveN(self.flow_var[(len(self.nodes),1)], index=0, priority=2, weight=1.0, 
                                             abstol=0.0, reltol=0.0, name="max_flow")
        
            # Secondary: Minimize edges used
            self.maxflow_model.setObjectiveN(grb.quicksum(self.edge_used[e] for e in self.mf_all_edges), index=1, priority=1, weight=-1.0,
                                             abstol=0.0, reltol=0.0, name="min_edges")
        elif routing_assumption == "distributed":
            # Primary: Maximize flow
            self.maxflow_model.setObjectiveN(self.flow_var[(len(self.nodes),1)], index=0, priority=3, weight=1.0, 
                                             abstol=0.0, reltol=0.0, name="max_flow")

            # Secondary: Maximize edges used
            self.maxflow_model.setObjectiveN(grb.quicksum(self.edge_used[e] for e in self.mf_all_edges), index=1, priority=2, weight=1.0,
                                             abstol=0.0, reltol=0.0, name="max_edges")
            
            # Tertiary: Minimize the number of edges used
            self.maxflow_model.setObjectiveN(grb.quicksum(self.flow_var[e] for e in self.mf_all_edges), index=1, priority=1, weight=-1.0,
                                             abstol=0.0, reltol=0.0, name="max_edges")
        elif routing_assumption == "least_vulnerable":
            # Primary: Maximize flow
            self.maxflow_model.setObjectiveN(self.flow_var[(len(self.nodes),1)], index=0, priority=2, weight=1.0, 
                                             abstol=0.0, reltol=0.0, name="max_flow")
        
            # Secondary: Minimize vulnerability (weighted by interdiction probability)
            self.maxflow_model.setObjectiveN(grb.quicksum(self.state["edge_interdiction_probability"][ind]*
                                                          (self.flow_var[e]+self.flow_var[(e[1],e[0])]) 
                                                          for ind, e in enumerate(self.interdictable_edges)), index=1, priority=1, weight=-1.0,
                                             abstol=0.0, reltol=0.0, name="least_vulnerable")
        else:
            raise ValueError(f"Unknown routing assumption: {routing_assumption}")
    
        self.old_routing = routing_assumption
    
    # BEGIN Gymnasium Environment Methods        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state and return observation."""
        # Clean up any existing models
        self._cleanup_models()

        # Call parent reset and set random seeds
        super().reset(seed=seed)
        if seed is not None:
            self._set_random_seeds(seed)

        # Generate network parameters
        # Sample edge capacities
        raw_capacities = self.base_spaces['edge_capacity'].sample()[:self.actual_num_edges]
        edge_capacities = ((raw_capacities / 100.0) * (self.DEFAULT_TRAINING_EDGE_CAPACITY_RANGE[1]-self.DEFAULT_TRAINING_EDGE_CAPACITY_RANGE[0]) + self.DEFAULT_TRAINING_EDGE_CAPACITY_RANGE[0]).astype(int)
        
        # Sample edge costs and interdiction probabilities
        raw_costs = self.base_spaces['edge_costs'].sample()[:self.actual_num_edges]
        edge_costs = (((raw_costs) / 10) * (self.DEFAULT_TRAINING_EDGE_COST_RANGE[1]-self.DEFAULT_TRAINING_EDGE_COST_RANGE[0]) + self.DEFAULT_TRAINING_EDGE_COST_RANGE[0]).astype(int)
        
        #Sample interdiction probabilities based on deterministic setting
        if self.deterministic_outcomes:
            edge_interdiction_probabilities = np.ones(self.max_num_edges, dtype=np.float32)
        else:
            probs = self.base_spaces['edge_interdiction_probability'].sample()[:self.actual_num_edges]
            # Round to 0.25 increments for consistency
            sample_rounded = np.round(probs * 4)
            edge_interdiction_probabilities = (sample_rounded.astype(float) / 4)
    
        # Sample budget based on initial budget setting."""
        if self.initial_budget is not None:
            remaining_budget = np.array([self.initial_budget], dtype=int)
        else:
            budget_sample = self.base_spaces['budget'].sample()
            # Map from 0-100 to training budget range
            budget_range = self.DEFAULT_TRAINING_BUDGET_RANGE
            scaled_budget = ((budget_range[1] - budget_range[0]) * budget_sample[0] / 100) + budget_range[0]
            remaining_budget = np.array([round(scaled_budget)], dtype=int)
            
        network_params = {
            'capacities': edge_capacities, 
            'costs': edge_costs,          
            'probabilities': edge_interdiction_probabilities,
            'budget': remaining_budget
        }

        # Create base state
        base_state = self._create_base_state(network_params)

        # Add strategy-specific components
        self.state = self._add_strategy_components(base_state)

        # Calculate reference objective value for the attacker's strategy
        if self.attacker_strategy == 'zero_sum':
            self.reference_obj, self.reference_flows = self._compute_objective_and_flows()
        elif self.attacker_strategy == 'canalize':
            self.reference_obj, self.reference_flows = self._calculate_canalize_objective_and_flows()
        elif self.attacker_strategy == 'isolate':
            self.reference_obj, self.reference_flows = self._calculate_isolate_objective_and_flows()
        elif self.attacker_strategy == 'divert':
            _, self.reference_flows = self.solve_max_flow()
            from_flow = self._calculate_target_path_flow(self.reference_flows, 'divert_from_objective')
            to_flow = self._calculate_target_path_flow(self.reference_flows, 'divert_to_objective')
            self.reference_start_flows = (from_flow, to_flow)
            self.reference_obj = 0
        
        self.last_obj = self.reference_obj
        self.reference_budget = remaining_budget[0]

        self._cache_flow_array()

        return self.state, {}

    def _cleanup_models(self):
        """Clean up any existing Gurobi models to free resources."""
        models_to_cleanup = ['master_model', 'sub_model', 'optimal_stochastic_model', 'optimal_stochastic_model_IM']
    
        for model_name in models_to_cleanup:
            if hasattr(self, model_name):
                try:
                    getattr(self, model_name).dispose()
                except Exception:
                    pass  # Continue if dispose fails
                delattr(self, model_name)
    
        # Clean up related attributes
        cleanup_attrs = ['benders_cuts', 'stochastic_alpha', 'stochastic_beta', 'stochastic_source_sink_constr', 'stochastic_aabg_constr', 'stochastic_alpha_IM', 'stochastic_beta_IM', 'stochastic_source_sink_constr_IM', 'stochastic_aabg_constr_IM']
    
        for attr in cleanup_attrs:
            if hasattr(self, attr):
                delattr(self, attr)

        self.num_stochastic_scenarios = None
        self.num_stochastic_scenarios_IM = None

    def _set_random_seeds(self, seed):      
        """Set random seeds for reproducibility."""
        # Set seeds for base spaces
        spaces_to_seed = ['edge_capacity', 'edge_costs', 'edge_interdiction_probability', 'budget']
        for space_name in spaces_to_seed:
            self.base_spaces[space_name].seed(seed)
    
        # Set seed for strategy-specific spaces if they exist
        strategy = self.strategy_spaces.get(self.attacker_strategy, {})
        for objective_name, objective_space in strategy.items():
            objective_space.seed(seed)
    
        # Set Python's random module seed
        random.seed(seed)
    
        # Set NumPy random seed (if using numpy random functions)
        np.random.seed(seed)

    def _create_base_state(self, network_params):
        """Create the base state dictionary with common components."""
        # Update edge attributes in the episode graph
        for edge, cap, cost, prob in zip(self.interdictable_edges,
                                         network_params['capacities'],
                                         network_params['costs'],
                                         network_params['probabilities']):
            e = self.edges_episode[edge]
            e.capacity = cap
            e.interdiction_cost = cost
            e.interdiction_probability = prob

        # Pad all edge arrays to max_num_edges
        padded_capacities = np.zeros(self.max_num_edges, dtype=int)
        padded_capacities[:self.actual_num_edges] = network_params['capacities']
    
        padded_costs = np.zeros(self.max_num_edges, dtype=int)
        padded_costs[:self.actual_num_edges] = network_params['costs']
    
        padded_probabilities = np.zeros(self.max_num_edges, dtype=np.float32)
        padded_probabilities[:self.actual_num_edges] = network_params['probabilities']
        
        # Create node arrays
        departure_nodes = np.zeros(self.max_num_edges, dtype=int)
        arrival_nodes = np.zeros(self.max_num_edges, dtype=int)
        departure_nodes[:self.actual_num_edges] = np.array(self.edge_departures)
        arrival_nodes[:self.actual_num_edges] = np.array(self.edge_arrivals)

        # CREATE EXPLICIT PADDING MASK
        padding_mask = np.zeros(self.max_num_edges, dtype=np.float32)
        padding_mask[:self.actual_num_edges] = 1.0  # Mark valid edges as 1
    
        return {
            'edge_capacity': padded_capacities,  
            'edge_interdicted': np.zeros(self.max_num_edges, dtype=int),
            'edge_costs': padded_costs,  
            'edge_interdiction_probability': padded_probabilities,  
            'edge_departure_node': departure_nodes,
            'edge_arrival_node': arrival_nodes,
            'budget': network_params['budget'],
            'padding_mask': padding_mask  # NEW - explicit padding information
        }

    def _add_strategy_components(self, base_state):
        """Add strategy-specific components to the state."""
        strategy_handlers = {"zero_sum": lambda: base_state,
                             "canalize": self._add_canalize_components,
                             "isolate": self._add_isolate_components,
                             "divert": self._add_divert_components}
    
        handler = strategy_handlers.get(self.attacker_strategy)
        if handler is None:
            raise ValueError(f"Unknown attacker strategy: {self.attacker_strategy}")
    
        return handler() if self.attacker_strategy == "zero_sum" else handler(base_state)

    def _add_canalize_components(self, base_state):
        """Add canalize-specific objective to state."""
        path_edges = self._find_simple_path()
        canalize_objective = np.zeros(self.max_num_edges, dtype=int)
        
        # Create boolean mask for path edges
        edge_in_path = np.array([edge in path_edges or (edge[1], edge[0]) in path_edges for edge in self.interdictable_edges], dtype=bool)

        # Vectorized assignment
        canalize_objective[:len(edge_in_path)] = edge_in_path.astype(int)
#        #for edge_id, edge in enumerate(self.interdictable_edges):
#        #    if edge in path_edges or (edge[1], edge[0]) in path_edges:
#        #        canalize_objective[edge_id] = 1
        return {**base_state, 'canalize_objective': canalize_objective}

    def _add_isolate_components(self, base_state):
        """Add isolate-specific objective to state (edge-based, sink-connected only)."""
        # Create padded objective with at least 1 marked sink edge
        isolate_objective = np.zeros(self.max_num_edges, dtype=int)
        num_to_mark = np.random.randint(1, self.num_cached_sink_edge_indices + 1)
        marked_indices = np.random.choice(self.cached_sink_edge_indices, size=num_to_mark, replace=False)
        isolate_objective[marked_indices] = 1
    
        return {**base_state, 'isolate_objective': isolate_objective}

    def _add_divert_components(self, base_state):
        """Add divert-specific objectives to state."""
        # Temporarily set state for max flow calculation
        temp_state = {**base_state, 'divert_from_objective': np.zeros(self.max_num_edges),
                      'divert_to_objective': np.zeros(self.max_num_edges)}
        self.state = temp_state

        # Find max flow path
        _, flows = self.solve_max_flow()
        from_path = self._extract_max_flow_path(flows)
    
        # Find alternative path avoiding max flow path
        to_path = self._find_alternative_path(from_path)
    
        # Convert paths to objective arrays
        divert_from = np.zeros(self.max_num_edges, dtype=int)
        for e, edge in enumerate(self.interdictable_edges):
            if edge in from_path or (edge[1],edge[0]) in from_path:
                divert_from[e] = 1
        # Padded entries remain 0
    
        divert_to = np.zeros(self.max_num_edges, dtype=int)
        for e, edge in enumerate(self.interdictable_edges):
            if edge in to_path or (edge[1],edge[0]) in to_path:
                divert_to[e] = 1
        return {**base_state, 'divert_from_objective': divert_from, 'divert_to_objective': divert_to}

    def _find_simple_path(self):
        """Find a simple path from source to sink."""
        path_edges = []
        current_node = 1
        visited = {1}
        sink = self.sink_nodes[0]
    
        while current_node != sink:
            valid_edges = [e for e in self.edge_groups[current_node]['out'] if e[1] not in visited and e[1] >= current_node - 1]
        
            if not valid_edges:
                # Restart if stuck
                current_node = 1
                visited = {1}
                path_edges = []
                continue
        
            selected_edge = random.choice(valid_edges)
            path_edges.append(selected_edge)
            visited.add(selected_edge[1])
            current_node = selected_edge[1]
    
        return set(path_edges)

    def _extract_max_flow_path(self, flows):
        """Extract the path with maximum flow from flows dictionary."""
        from_path = set()
        current_node = 1
        sink = self.sink_nodes[0]
    
        while current_node != sink:
            outgoing_edges = self.edge_groups[current_node]['out']
            next_edge = max(outgoing_edges, key=lambda e: flows.get(e, 0))
            from_path.add(next_edge)
            current_node = next_edge[1]
    
        return from_path

    def _find_alternative_path(self, avoid_edges):
        """Find an alternative path avoiding specified edges."""
        path_edges = set()
        current_node = 1
        visited = {1}
        sink = self.sink_nodes[0]
    
        while current_node != sink:
            valid_edges = []
            for edge in self.edge_groups[current_node]['out']:
                target = edge[1]
                if (target not in visited and target >= current_node - 1 and 
                    edge not in avoid_edges):
                
                    # Check for valid future moves
                    if target != sink:
                        future_valid = any(
                            e not in avoid_edges and e[1] != current_node and e[1] >= e[0] - 1
                            for e in self.edge_groups[target]['out']
                        )
                        if not future_valid:
                            continue
                
                    valid_edges.append(edge)
        
            if not valid_edges:
                # Restart if no valid path
                current_node = 1
                visited = {1}
                path_edges = set()
                continue
        
            selected_edge = random.choice(valid_edges)
            path_edges.add(selected_edge)
            visited.add(selected_edge[1])
            current_node = selected_edge[1]
    
        return path_edges

    def step(self, action):  
        """Execute one step in the environment based on the given action."""
        # Initialize step variables
        done = False
        do_nothing = False
        remaining_budget = self.state['budget'].copy()

        # Determine if action was "do nothing"
        if action == self.max_num_edges:
            self.state['budget'] = np.array([0])
            reward = 0
            return self.state, float(reward), True, False, {}

        # Validate action
        valid_action = self._validate_action(action, remaining_budget, self.state['edge_interdicted'])
        
        # Apply action effects
        if valid_action:
            #Determine reward and decrement budget
            remaining_budget = self._apply_valid_action(action, remaining_budget)
            #Compute Rewards
            strategy_calculators = {"zero_sum": self._calculate_zero_sum_reward,
                                    "canalize": self._calculate_canalize_reward,
                                    "isolate": self._calculate_isolate_reward,
                                    "divert": self._calculate_divert_reward}
            calculator = strategy_calculators.get(self.attacker_strategy)
            reward = calculator()
        else:
            #Determine penalty and decrement budget
            reward, remaining_budget = self._apply_invalid_action(action, remaining_budget)

        # Check if episode is complete
        done = self._is_episode_complete(remaining_budget)
    
        # Update state
        self.state['budget'] = remaining_budget
    
        return self.state, float(reward), bool(done), False, {}
        
    def _validate_action(self, action, remaining_budget, interdicted_edges):
        """Validate if the given action is legal."""
        ## Checks for all attacker strategies
        # Check if action is within action space
        if action >= self.actual_num_edges:  # Padded actions are invalid
            return False
    
        # Check budget constraint
        if remaining_budget[0] - self.state['edge_costs'][action] < -0.1:
            return False
    
        # Check capacity constraint
        if self.state['edge_capacity'][action] == 0:
            return False

        # Check interdiction probability constraint
        if self.state['edge_interdiction_probability'][action] == 0:
            return False
    
        # Check interdiction limit
        max_interdictions = self.MAX_INTERDICTION_ATTEMPTS if self.multiple_interdiction_attempts else 1
        if interdicted_edges[action]+1 > max_interdictions:
            return False

        ## Attacker Strategy Specific Checks
        # Zero-Sum - Check target has previous flow
    #    if self.attacker_strategy == 'zero_sum':
        edge = self.interdictable_edges[action]
        if self.reference_flows[edge] == 0 and self.reference_flows[(edge[1],edge[0])] == 0:
            return False
        
        # Canalization - Check attacker does not target canalization path
        if self.attacker_strategy == 'canalize':
            if self.state['canalize_objective'][action] == 1:
                return False

        # Divert - Check attacker does not target the divert to path
        if self.attacker_strategy == 'divert':
            if self.state['divert_to_objective'][action] == 1:
                return False
        
        return True
        
    def _apply_valid_action(self, action, remaining_budget):
        """Apply the effects of a valid action."""
        # Deduct cost from budget
        remaining_budget[0] = remaining_budget[0] - self.state['edge_costs'][action]
    
        # Mark edge as interdicted
        self.state['edge_interdicted'][action] += 1
    
        return remaining_budget

    def _apply_invalid_action(self, action, remaining_budget):
        """Apply penalty for invalid action."""
        penalty = self.PENALTY_VALUE
        remaining_budget[0] = max(0, remaining_budget[0] - self.state['edge_costs'][action])
        return penalty, remaining_budget

    def _calculate_zero_sum_reward(self):
        """Calculate reward for zero-sum strategy (maximize disruption)."""
        objective_value, self.reference_flows = self._compute_objective_and_flows()
        reward = max(self.last_obj - objective_value, 0) / self.reference_budget
        if reward > 0:
            self.last_obj = objective_value   
        elif reward ==0:
            reward = self.PENALTY_VALUE
        return reward

    def _calculate_stochastic_objective_and_flow(self, strategy_type="zero_sum", use_optimized=False):
        """
        Calculate stochastic objective and flows using Monte Carlo sampling.
        
        Parameters:
        -----------
        strategy_type : str
            Type of attacker strategy ('zero_sum', 'canalize', 'isolate', 'divert')
        use_optimized : bool
            If True, uses optimized approach (group outcomes and solve once per unique state)
            If False, uses legacy approach (solve for each sample)
    
        Returns:
        --------
        tuple: (mean_objective, mean_flows)
        """
    
        if use_optimized:
            return self._calculate_stochastic_optimized(strategy_type)
        else:
            return self._calculate_stochastic_legacy(strategy_type)

    def _calculate_stochastic_optimized(self, strategy_type="zero_sum"):
        """
        Optimized stochastic calculation: group by unique outcomes and weight by probability.
    
        This method:
        1. Samples interdiction outcomes (success/failure) based on probabilities
        2. Groups identical outcomes together
        3. Solves max flow once per unique outcome
        4. Computes weighted average based on outcome frequencies
        """
        # Extract interdiction probabilities
        probs = self.state['edge_interdiction_probability'][:self.num_interdictable_edges]
        interdicted = self.state['edge_interdicted'][:self.num_interdictable_edges].astype(int)
    
        # Sample interdiction outcomes
        outcome_samples = []
        for _ in range(self.SAMPLE_SIZE):
            # For each edge, determine if interdiction succeeds based on probability
            
            # Create outcome tuple (which edges are successfully interdicted)
            if self.multiple_interdiction_attempts:
                # Add to existing interdiction count
                failure_probs = ((1 - probs) ** interdicted)
                success = np.random.binomial(1, 1-failure_probs)
                outcome = tuple(np.minimum(interdicted, success)) #tuple(interdicted + success)
            else:
                success = np.random.binomial(1, probs)
                # Binary: either interdicted or not
                outcome = tuple(np.minimum(interdicted, success))
        
            outcome_samples.append(outcome)
    
        # Count unique outcomes and their frequencies
        outcome_counts = Counter(outcome_samples)
        unique_outcomes = list(outcome_counts.keys())
    
 #       print(f"  Optimized stochastic: {len(unique_outcomes)} unique outcomes from {self.SAMPLE_SIZE} samples")
    
        # Solve max flow once per unique outcome
        outcome_results = {}
        for outcome in unique_outcomes:
            # Convert outcome to capacity dict
            capacity_dict = self._create_capacity_dict_from_outcome(outcome)
        
            # Solve max flow for this outcome
            obj, flows = self.solve_max_flow(capacity_dict, list(outcome))
        
            # Calculate strategy-specific objective
            if strategy_type == "zero_sum":
                objective = obj
            elif strategy_type == "canalize":
                objective = self._calculate_target_path_flow(flows, 'canalize_objective')
            elif strategy_type == "isolate":
                objective = self._calculate_target_edge_flow(flows, 'isolate_objective')
            elif strategy_type == "divert":
                from_flow = self._calculate_target_path_flow(flows, 'divert_from_objective')
                to_flow = self._calculate_target_path_flow(flows, 'divert_to_objective')
                diverted_flow_from = self.reference_start_flows[0] - from_flow
                diverted_flow_to = to_flow - self.reference_start_flows[1]
                objective = np.min([diverted_flow_from, diverted_flow_to])
        
            outcome_results[outcome] = {
                'objective': objective,
                'flows': flows,
                'count': outcome_counts[outcome]
            }
        # Compute weighted averages
        total_samples = sum(outcome_counts.values())
        weighted_objective = sum(
            result['objective'] * result['count'] / total_samples
            for result in outcome_results.values()
        )
    
        # Compute weighted average flows
        weighted_flows = defaultdict(float)
        for outcome, result in outcome_results.items():
            weight = result['count'] / total_samples
            for edge, flow in result['flows'].items():
                weighted_flows[edge] += flow * weight
    
        return weighted_objective, dict(weighted_flows)


    def _calculate_stochastic_legacy(self, strategy_type="zero_sum"):
        """
        Legacy Calculate objective value under stochastic interdiction outcomes.
        
        Args:
            strategy_type: One of 'zero_sum', 'canalize', 'isolate', or 'divert'
        
        Returns:
            mean_objective: Mean objective across all Monte Carlo iterations
        """
        if self.multiple_interdiction_attempts:
            edges_interdicted = (self.state['edge_interdicted'] > 0).astype(int)
            success_probs = ((1 - self.state['edge_interdiction_probability']) ** 
                            self.state['edge_interdicted'])
            prob_array = edges_interdicted * success_probs
        else:
            prob_array = (self.state['edge_interdicted'] * 
                         self.state['edge_interdiction_probability'])

        non_zero_probs = prob_array[prob_array != 0]
        if non_zero_probs.size == 0:
            iterations = 1
        else:
            mean_prob = np.mean(non_zero_probs)
            if mean_prob <= 0.5:
                iterations = int(1 + (1000 - 1) * (mean_prob / 0.5))
            else:
                iterations = int(1000 - (1000 - 1) * ((mean_prob - 0.5) / 0.5))
        
        objectives = []
        all_flows = []  # Store flows from each iteration

    
        for _ in range(iterations):
            # Solve max flow for this iteration
            max_flow_obj, flows = self.solve_max_flow()
            all_flows.append(flows)  # Store the flows
        
            # Compute the objective based on strategy type
            if strategy_type == 'zero_sum':
                objective = max_flow_obj
            
            elif strategy_type == 'canalize':
                objective = self._calculate_target_path_flow(flows, 'canalize_objective')
            
            elif strategy_type == 'isolate':
                objective = self._calculate_target_edge_flow(flows, 'isolate_objective')
            
            elif strategy_type == 'divert':
                from_flow = self._calculate_target_path_flow(flows, 'divert_from_objective')
                to_flow = self._calculate_target_path_flow(flows, 'divert_to_objective')
                
                diverted_flow_from = self.reference_start_flows[0] - from_flow
                diverted_flow_to = to_flow - self.reference_start_flows[1] 
                objective = np.min([diverted_flow_from,diverted_flow_to])
            
            objectives.append(objective)

        # Compute mean flows across all iterations
        mean_flows = {}
        if all_flows:
            # Get all unique edges across all iterations
            all_edges = set()
            for flow_dict in all_flows:
                all_edges.update(flow_dict.keys())
        
            # Compute mean for each edge
            for edge in all_edges:
                edge_flows = [flow_dict.get(edge, 0) for flow_dict in all_flows]
                mean_flows[edge] = np.mean(edge_flows)
        
        # Return mean objective and mean flows
        return np.mean(objectives), mean_flows

    def _create_capacity_dict_from_outcome(self, interdiction_outcome):
        """
        Create capacity dictionary from interdiction outcome.
    
        Parameters:
        -----------
        interdiction_outcome : tuple or array
            Interdiction state for each interdictable edge
    
        Returns:
        --------
        dict: Mapping from edge to remaining capacity after interdictions
        """
        capacity_dict = {}
    
        for idx, edge in enumerate(self.interdictable_edges):
            base_capacity = self.state['edge_capacity'][idx]
        
            if self.multiple_interdiction_attempts:
                # Each interdiction reduces capacity
                is_interdicted = interdiction_outcome[idx]
                remaining_capacity = 0 if is_interdicted else base_capacity
            else:
                # Binary: either full capacity or zero
                is_interdicted = interdiction_outcome[idx]
                remaining_capacity = 0 if is_interdicted else base_capacity
        
            capacity_dict[edge] = remaining_capacity
    
        # Add non-interdictable edges with full capacity
        for edge in self.edges_reset:
            if edge not in self.interdictable_edges:
                capacity_dict[edge] = self.state['edge_capacity'][idx]
    
        return capacity_dict



    
    def _compute_objective_and_flows(self, deterministic_mode=None):
        """Calculate the max flow objective and edge flows."""
        if deterministic_mode is None:
            deterministic_mode = self.deterministic_outcomes
        
        if deterministic_mode:
            objective, flows = self.solve_max_flow()
        else:
            # Stochastic outcome calculation
            objective, flows = self._calculate_stochastic_objective_and_flow('zero_sum', True)
    
        return objective, flows

    def _calculate_canalize_objective_and_flows(self):
        """Calculate objective for canalize strategy (flow through specific path)."""
        if self.deterministic_outcomes:
            _, flows = self.solve_max_flow()
            target_path_flow = self._calculate_target_path_flow(flows, 'canalize_objective')
            return target_path_flow, flows
        else:
            # Stochastic calculation - returns mean objective directly
            objective, mean_flows = self._calculate_stochastic_objective_and_flow('canalize', True)
            return objective, mean_flows
        
    def _calculate_canalize_reward(self):
        """Calculate reward for canalize strategy (force flow through specific path)."""
        # Reward for successful interdiction of non-target edges
        target_path_flow, self.reference_flows = self._calculate_canalize_objective_and_flows()
        
        reward = (target_path_flow - self.last_obj) / self.reference_budget
        self.last_obj = target_path_flow
        if reward == 0:
            reward = self.PENALTY_VALUE
        return reward
        
    def _calculate_isolate_objective_and_flows(self):
        """Calculate objective for isolate strategy (reduce flow on specific edges)."""
        if self.deterministic_outcomes:
            _, flows = self.solve_max_flow()
            target_node_flow = self._calculate_target_edge_flow(flows, 'isolate_objective')
            return target_node_flow, flows
        else:
            # Stochastic calculation - returns mean objective directly
            objective, mean_flows = self._calculate_stochastic_objective_and_flow('isolate',True)
            return objective, mean_flows
        
    def _calculate_isolate_reward(self):
        """Calculate reward for isolate strategy (reduce flow to specific nodes)."""
        # Reward reduction in flow to target nodes
        target_node_flow, self.reference_flows = self._calculate_isolate_objective_and_flows()
        
        reward = (self.last_obj-target_node_flow) / self.reference_budget
        self.last_obj = target_node_flow
        if reward == 0:
            reward = self.PENALTY_VALUE
        return reward

    def _calculate_divert_objective_and_flows(self, mode = None):
        """Calculate objective for divert strategy (redirect flow from one path to another)."""
        if mode is None:
            mode = self.deterministic_outcomes

        if mode:
            _, flows = self.solve_max_flow()
            from_flow = self._calculate_target_path_flow(flows, 'divert_from_objective')
            to_flow = self._calculate_target_path_flow(flows, 'divert_to_objective')
            diverted_flow_from = self.reference_start_flows[0] - from_flow
            diverted_flow_to = to_flow - self.reference_start_flows[1] 
            objective = np.min([diverted_flow_from,diverted_flow_to])
            
            return objective, flows
        else:
            # Stochastic calculation - returns mean objectives directly
            mean_objective, mean_flows = self._calculate_stochastic_objective_and_flow('divert',True)
            # Return as tuple to maintain consistent interface with reward calculation
            return mean_objective, mean_flows

    def _calculate_divert_reward(self):
        """Calculate reward for divert strategy (redirect flow from one path to another)."""
        # Calculate reward based on flow diversion success
        diverted_flow, self.reference_flows = self._calculate_divert_objective_and_flows()
        
        reward = (diverted_flow - self.last_obj) / self.reference_budget
        self.last_obj = diverted_flow
        if reward == 0:
            reward = self.PENALTY_VALUE
        return reward

    def _calculate_target_path_flow(self, flows, objective_key):
        """Calculate total flow through edges marked in the objective."""
        objective = self.state[objective_key]
        target_flows = []
    
        for idx, edge in enumerate(self.interdictable_edges):
            if objective[idx] == 1:
                forward_flow = flows.get(edge, 0)
                reverse_flow = flows.get((edge[1], edge[0]), 0)
                target_flows.append(forward_flow+reverse_flow)
    
        # Return minimum flow among target edges
        return min(target_flows)

    def _calculate_target_edge_flow(self, flows, objective_key):
        """Calculate total flow on edges marked in the objective."""
        objective = self.state[objective_key]
        # Get indices where objective is 1
        target_indices = np.where(objective[:self.actual_num_edges] == 1)[0]
        
        total_flow = 0

        target_edges = [self.interdictable_edges[i] for i in target_indices]
    
        # Batch get flows
        flows_array = np.array([(flows.get(edge, 0) + flows.get((edge[1], edge[0]), 0)) for edge in target_edges])
    
        # Return sum flow among target nodes
        return np.sum(flows_array) #np.sum(total_flow)

    def _is_episode_complete(self, remaining_budget):
        """Determine if the episode should end."""
        # Calculate minimum resources needed for next action
        if self.multiple_interdiction_attempts:
            least_resources = min(self.state['edge_costs'][self.state['edge_costs'] > 0], default=float('inf'))
        else:
            available_costs = self.state['edge_costs'] * (1 - self.state['edge_interdicted'])
            least_resources = min(available_costs[available_costs > 0], default=float('inf'))
    
        # Episode ends if insufficient budget or network is completely disrupted
        if remaining_budget[0] < least_resources:
            return True
    
        if self.deterministic_outcomes:
            objective_value, _ = self.solve_max_flow()
            if objective_value == 0:
                return True
        return False

    def render(self, mode='human'):
        """
        Render the environment state, displaying only values where padding_mask == 1.
        """
        if mode != "human":
            return
    
        print("=" * 80)
        print("ENVIRONMENT STATE (Non-padded values only)")
        print("=" * 80)
    
        # Get padding mask
        padding_mask = self.state.get("padding_mask", np.ones(self.max_num_edges))
        valid_indices = np.where(padding_mask == 1)[0]
    
        print(f"\nNumber of valid (non-padded) edges: {len(valid_indices)} / {self.max_num_edges}")
        print(f"Actual number of edges in graph: {self.actual_num_edges}")
    
        # Budget information
        print(f"\n{'Budget Information':^80}")
        print("-" * 80)
        print(f"Remaining Budget: {self.state['budget'][0]}")
        print(f"Reference Budget: {self.reference_budget}")
    
        # Edge-based state information (filtered by padding mask)
        print(f"\n{'Edge State Information':^80}")
        print("-" * 80)
        print(f"{'Index':<8} {'Origin':<8} {'Dest':<8} {'Capacity':<12} {'Cost':<8} {'Int. Prob':<12} {'Interdicted':<12}")
        print("-" * 80)
    
        for idx in valid_indices[:25]:  # Show first 25 valid edges
            capacity = self.state['edge_capacity'][idx]
            cost = self.state['edge_costs'][idx]
            prob = self.state['edge_interdiction_probability'][idx]
            interdicted = self.state['edge_interdicted'][idx]
            departure_node = self.state['edge_departure_node'][idx]
            arrival_node = self.state['edge_arrival_node'][idx]
        
            print(f"{idx:<8} {departure_node:<8} {arrival_node:<8} {capacity:<12} {cost:<8} {prob:<12.2f} {interdicted:<12}")
    
        if len(valid_indices) > 25:
            print(f"... ({len(valid_indices) - 25} more valid edges)")
    
        # Strategy-specific objectives (if present)
        strategy_objectives = {
            'canalize': 'canalize_objective',
            'isolate': 'isolate_objective',
            'divert_from': 'divert_from_objective',
            'divert_to': 'divert_to_objective'
        }
    
        has_objectives = any(obj_key in self.state for obj_key in strategy_objectives.values())
    
        if has_objectives:
            print(f"\n{'Strategy Objectives':^80}")
            print("-" * 80)
        
            for strategy_name, obj_key in strategy_objectives.items():
                if obj_key in self.state:
                    objective_values = self.state[obj_key][valid_indices]
                    num_target_edges = np.sum(objective_values == 1)
                    print(f"{strategy_name.capitalize()}: {num_target_edges} target edges (out of {len(valid_indices)} valid)")
    
        # Additional state fields (non-edge specific)
        print(f"\n{'Other State Information':^80}")
        print("-" * 80)
    
        if 'episode_step' in self.state:
            print(f"Episode Step: {self.state['episode_step'][0]}")
    
        if 'max_flow_value' in self.state:
            print(f"Max Flow Value: {self.state['max_flow_value'][0]:.2f}")
    
        print(f"\nReference Objective: {self.reference_obj}")
    
        if hasattr(self, 'last_obj'):
            print(f"Last Objective: {self.last_obj}")
    
        print("=" * 80)
        # END Gymnasium Environment Methods
            
    def solve_optimal_interdiction(self):
        if self.deterministic_outcomes == True: #Solve Deterministic Case with Wood's Max/Min Formulation
            if not hasattr(self, 'optimal_deterministic_model'):
                # Initialize the Gurobi model
                self.optimal_deterministic_model = grb.Model("Network Interdiction Model 1U", env=self.GUROBI_ENV)
                
                # Define Decision Variables
                self.alpha = self.optimal_deterministic_model.addVars(self.nodes.keys(), vtype=grb.GRB.BINARY, name="alpha")
                self.beta = self.optimal_deterministic_model.addVars(self.edges_with_sink_source, vtype=grb.GRB.BINARY, name="beta")
                self.gamma = self.optimal_deterministic_model.addVars(self.interdictable_edges, vtype=grb.GRB.BINARY, name="gamma")
                
                # Define Constraints
                self.optimal_deterministic_model.addConstrs(
                    (self.alpha[e[0]] - self.alpha[e[1]] + self.beta[e] + 
                     (self.gamma[e] if e in self.interdictable_edges else 0) >= 0 for e in self.edges_reset.keys()), name="flow_conservation")

                self.optimal_deterministic_model.addConstrs(
                    (self.alpha[e[1]] - self.alpha[e[0]] + self.beta[e] + 
                     (self.gamma[e] if e in self.interdictable_edges else 0) >= 0 for e in self.edges_reset.keys()),
                    name="flow_conservation_reverse")

                self.optimal_deterministic_model.addConstr(self.alpha[self.sink_nodes[0]]-self.alpha[self.source_nodes[0]]+self.beta[(self.sink_nodes[0],self.source_nodes[0])] >=1,
                                                          name = "sink-source")
            
            # Update Constraints
            if hasattr(self, 'budget_constr'):
                self.optimal_deterministic_model.remove(self.budget_constr)

            self.budget_constr = self.optimal_deterministic_model.addConstr(
                grb.quicksum(self.edges_episode[e].interdiction_cost * self.gamma[e]
                             for e in self.interdictable_edges) <= self.state['budget'][0],
                name="budget"
            )

            # Define Objective Value
            self.optimal_deterministic_model.setObjective(grb.quicksum(edge.capacity * self.beta[edge_id] for edge_id, edge in self.edges_episode.items())+self.MAX_SOURCE_FLOW*self.beta[(self.sink_nodes[0],self.source_nodes[0])], grb.GRB.MINIMIZE)

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
            self.optimal_stochastic_model = grb.Model("Stochastic Model", env=self.GUROBI_ENV)

            # Creating decision variables
            self.stochastic_gamma = self.optimal_stochastic_model.addVars(self.interdictable_edges, vtype=grb.GRB.BINARY, name="gamma")

            # Create Variable Lower Bounds
            self.optimal_stochastic_model.setAttr("LB", [self.stochastic_gamma[e] for e in interdicted_edges],1)
            
             # Budget constraint
            self.stochastic_budget_constr = self.optimal_stochastic_model.addConstr(grb.quicksum(
                self.edges_episode[e].interdiction_cost * self.stochastic_gamma[e] 
                for e in self.interdictable_edges) <= self.state['budget'][0], name="budget")

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
            self.stochastic_beta = self.optimal_stochastic_model.addVars([(e, s) for s in self.scenarios for e in self.edges_with_sink_source],
                                                                          vtype=grb.GRB.BINARY, name="beta")

            if hasattr(self, 'stochastic_source_sink_constr'):
                self.optimal_stochastic_model.remove(self.stochastic_source_sink_constr)
                del self.stochastic_source_sink_constr 

            self.stochastic_source_sink_constr = self.optimal_stochastic_model.addConstrs((self.stochastic_alpha[self.sink_nodes[0],s] - self.stochastic_alpha[self.source_nodes[0], s] +self.stochastic_beta[(self.sink_nodes[0],self.source_nodes[0]),s]>= 1 for s in self.scenarios), name="source_sink")

            # Objective Function
            self.optimal_stochastic_model.setObjective((1/n_scenarios)*grb.quicksum(self.edges_episode[e].capacity * self.stochastic_beta[e, s]
                for s in self.scenarios
                for e in self.edges_reset) + grb.quicksum(self.MAX_SOURCE_FLOW*self.stochastic_beta[(self.sink_nodes[0],self.source_nodes[0]),s] for s in self.scenarios), grb.GRB.MINIMIZE)
        
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
            self.optimal_stochastic_model_IM = grb.Model("Stochastic Model_IM", env=self.GUROBI_ENV)

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
                for e in self.interdictable_edges for k in range(1,9)) <= self.state['budget'][0], name="budget_IM")

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

    def solve_backward_induction(self, verbose=False):
        """
        Solve the optimal interdiction strategy for attacker using backward induction.
        This method finds the optimal interdictions for a particular attacker strategy.
    
        Returns:
            tuple: (optimal_objective_value, optimal_interdiction_sequence)
        """
        
        # Calculate minimum edge cost for depth estimation (only from real edges)
        real_edge_costs = self.state['edge_costs'][:self.actual_num_edges]
        self.min_edge_cost = min(real_edge_costs[real_edge_costs > 0], default=float('inf'))
        if self.min_edge_cost == float('inf'):
            self.min_edge_cost = 1  # Fallback
    
        # Estimate total states
        max_budget = self.state['budget'][0]
        budget_levels = max_budget // self.min_edge_cost
        estimated_states = self.actual_num_edges ** budget_levels  # Use actual_num_edges, not max
        update_rate = max(estimated_states // 20, 1)
        
        # Memoization dictionary: state -> (max_reward, best_action_sequence)
        memo = {}

        states_processed = 0
        pbar = tqdm(total=estimated_states, desc="DP States", unit=" states", disable=not verbose)

        # Initialize the dynamic programming table
        def state_to_key(interdicted_state):
            return tuple(interdicted_state)
        
        def update_progress(num_states_processed):
            # Update progress every 100 states
            nonlocal states_processed
            states_processed += num_states_processed
            if states_processed>=update_rate:
                pbar.update(states_processed)
                pbar.set_postfix({'Memo': len(memo)})
                states_processed = 0
            return
        
        def dp_solve(remaining_budget, interdicted_state, depth=0):
            """Dynamic programming recursive function for backward induction."""           
            state_key = state_to_key(interdicted_state)
        
            # Check if we've already solved this state
            if state_key in memo:
                update_progress(self.actual_num_edges**(budget_levels-depth))
                return memo[state_key]

            temp_state = self.state.copy()
            temp_state['edge_interdicted'] = interdicted_state.copy()
            temp_state['budget'] = np.array([remaining_budget])
                
            old_state = self.state
            self.state = temp_state

            if self.attacker_strategy == "zero_sum":
                final_objective, self.reference_flows = self._compute_objective_and_flows()
                final_objective = -final_objective
            elif self.attacker_strategy == 'canalize':
                final_objective, self.reference_flows = self._calculate_canalize_objective_and_flows()
            elif self.attacker_strategy == 'isolate':
                final_objective, self.reference_flows = self._calculate_isolate_objective_and_flows()
                final_objective = -final_objective
            elif self.attacker_strategy == 'divert':
                final_objective, self.reference_flows = self._calculate_divert_objective_and_flows()
                
            self.state = old_state
            
            # Base case: no more budget or maximum depth reached
            if remaining_budget < self.min_edge_cost or depth >= 20:
                               
                memo[state_key] = (final_objective, [])
                update_progress(self.actual_num_edges**(budget_levels-depth))
                return final_objective, []
       
            # Find all valid actions from current state
            valid_actions = []
            for action in range(self.actual_num_edges):
                edge = self.interdictable_edges[action]
                if self._validate_action(action, [remaining_budget], interdicted_state): # and (flows.get(edge, 0) != 0 or flows.get((edge[1],edge[0]), 0) != 0):
                    valid_actions.append(action)
                else:
                    update_progress(self.actual_num_edges**(budget_levels-(depth+1)))
        
            # If no valid actions, evaluate terminal state
            if not valid_actions:
                memo[state_key] = (final_objective, [])
                update_progress(self.actual_num_edges**(budget_levels-depth))
                return final_objective, []
        
            # Evaluate each possible action
            best_reward = -float('inf')
            best_sequence = []
        
            for action in valid_actions:
                # Create new state after taking this action
                new_interdicted_state = interdicted_state.copy()
                new_interdicted_state[action] += 1
                new_budget = remaining_budget - self.state['edge_costs'][action]
            
                # Recursively solve for the remaining problem
                future_reward, future_sequence = dp_solve(new_budget, new_interdicted_state, depth + 1)
            
                # Update best solution if this is better
                if future_reward > best_reward:
                    best_reward = future_reward
                    best_sequence = [action] + future_sequence
        
            memo[state_key] = (best_reward, best_sequence)
            return best_reward, best_sequence

        
        # Start the backward induction from current state
        initial_budget = self.state['budget'][0]
        initial_interdicted_state = self.state['edge_interdicted'].copy()
    
        optimal_reward, optimal_sequence = dp_solve(initial_budget, initial_interdicted_state)
        pbar.update(states_processed)        
        pbar.close()
        
        if self.attacker_strategy == "zero_sum" or self.attacker_strategy == 'isolate':
            optimal_reward = -optimal_reward
        
        optimal_actions = [self.interdictable_edges[idx] for idx in optimal_sequence]

        return optimal_reward, optimal_actions