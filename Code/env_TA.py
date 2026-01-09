#Purpose: Gymnasium environment for single/multiple attempt max flow deterministic/stochastic network interdiction by zero-sum/threat adaptive attackers

# Import all required packages
import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress most logs (including CUDA errors)
os.environ["RAY_DISABLE_USAGE_STATS"] = "1"
os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import pandas as pd
import gurobipy as grb                # Gurobi optimization library for solving mathematical models
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy, random
from tqdm import tqdm
#import tensorflow as tf
#tf.get_logger().setLevel('ERROR')          # Optional: Suppress Python-

from collections import defaultdict, Counter
from itertools import product

import ray

# Reduce native logging noise (best-effort; affects Python loggers)
import logging
logging.getLogger("ray").setLevel(logging.WARNING)
logging.getLogger("raylet").setLevel(logging.WARNING)

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
    GUROBI_ENV = grb.Env(params={"OutputFlag": 0, "LogToConsole": 0, "Threads": 1, "Seed": 1})
    
    def __init__(self, nodes, edges, deterministic_agent=True, initial_budget = None, 
                 multiple_interdiction_attempts=True, attacker_strategy="zero_sum",
                 budget_range=(0, 100), edge_capacity_range=(0, 100), 
                 edge_cost_range=(0, 10), training_budget_range=(5, 10), 
                 training_edge_capacity_range=(30, 60), training_edge_cost_range=(3, 5),
                 max_interdiction_attempts=10, max_source_flow=5, 
                 max_sink_need=5, penalty_value=-0.1, 
                 sample_size=1000, max_path_length = 6,
                 max_num_edges=500, 
                 max_num_nodes=250, old_routing="none", outcome_memo_actor=None, outcome_memo_actors=None):
        super(CustomEnv, self).__init__()

        #Setup core environment attributes
        self.nodes = nodes
        self.edges_reset = edges
        self.edges_episode = copy.deepcopy(self.edges_reset)
        self.deterministic_outcomes = deterministic_agent
        self.initial_budget = initial_budget
        self.multiple_interdiction_attempts = multiple_interdiction_attempts
        self.attacker_strategy = attacker_strategy
        self.DEFAULT_BUDGET_RANGE = budget_range 
        self.DEFAULT_EDGE_CAPACITY_RANGE = edge_capacity_range 
        self.DEFAULT_EDGE_COST_RANGE = edge_cost_range 
        self.DEFAULT_TRAINING_BUDGET_RANGE = training_budget_range 
        self.DEFAULT_TRAINING_EDGE_CAPACITY_RANGE = training_edge_capacity_range 
        self.DEFAULT_TRAINING_EDGE_COST_RANGE = training_edge_cost_range 
        self.MAX_INTERDICTION_ATTEMPTS = max_interdiction_attempts 
        self.MAX_SOURCE_FLOW = max_source_flow 
        self.MAX_SINK_NEED = max_sink_need 
        self.PENALTY_VALUE = penalty_value 
        self.SAMPLE_SIZE = sample_size 
        self.MAX_PATH_LENGTH = max_path_length
        self.max_num_edges = max_num_edges
        self.max_num_nodes = max_num_nodes
        self.old_routing = old_routing
        self.outcome_memo_actor = outcome_memo_actor
        self.outcome_memo_actors = outcome_memo_actors
        if self.outcome_memo_actors is None and self.outcome_memo_actor is not None:
             self.outcome_memo_actors = [self.outcome_memo_actor]
        self.local_outcome_cache = {} # Add local cache here

        self.num_stochastic_scenarios = None
        self.num_stochastic_scenarios_IM = None

        self.max_interdictions = self.MAX_INTERDICTION_ATTEMPTS if self.multiple_interdiction_attempts else 1
        
        # Initialize network structure
        self._setup_network_structure()

        # Setup observation and action spaces
        self._setup_spaces()

    def _cache_flow_array(self):
        """Fully vectorized cache using array indexing."""
        # Optimized to use pre-computed reverse keys
        flows = self.reference_flows
        
        self.cached_flow_array = np.array(
            [flows.get(e, 0) + flows.get(re, 0) for e, re in zip(self.both_edges, self.reverse_edges_list)], 
            dtype=np.float32
        )
    
    def _setup_network_structure(self):
        """Initialize network nodes and edges structure."""
        # Define node types
        self.super_source_nodes = [1]
        self.super_sink_nodes = [self.max_num_nodes]
        self.intermediate_nodes = list(range(2, len(self.nodes)))
              
        # Extract interdictable edges and their attributes
        self.both_edges = []
        self.interdictable_edges =[]
        self.noninterdictable_edges = []
        self.edge_departures =[]
        self.edge_arrivals =[]
        
        for key, edge in self.edges_reset.items():
            self.both_edges.append(key)
            if edge.interdictable == 1:
                self.interdictable_edges.append(key)
            else:
                self.noninterdictable_edges.append(key)
            self.edge_departures.append(key[0])
            self.edge_arrivals.append(key[1])

        self.all_interdictable_edges = list(self.interdictable_edges) + [(v, u) for (u, v) in self.interdictable_edges]
        self.all_noninterdictable_edges = list(self.noninterdictable_edges) + [(v, u) for (u, v) in self.noninterdictable_edges]

        #Create all possible edges
        self.all_both_edges = self.all_interdictable_edges + self.all_noninterdictable_edges
        
        # Create edge groups for efficient lookup
        out_edges = defaultdict(list)
        in_edges = defaultdict(list)
        for edge in self.all_both_edges:
            out_edges[edge[0]].append(edge)
            in_edges[edge[1]].append(edge)

        self.edge_groups ={node_id: {
            'out': out_edges.get(node_id, []),
            'in': in_edges.get(node_id, [])}
                           for node_id in self.nodes}
        
        # Create edge-to-index mapping
        self.edge_to_index = {edge: idx for idx, edge in enumerate(self.both_edges)}  #Changed from interdictable to both_edges

        # Create all three index lists in a single pass
        self.super_source_out_indices = []
        self.super_sink_in_indices = []
        self.noninterdictable_indices = []

        for idx, edge in enumerate(self.both_edges):
            if edge[0] == self.super_source_nodes[0]:
                self.super_source_out_indices.append(idx)
            if edge[1] == self.super_sink_nodes[0]:
                self.super_sink_in_indices.append(idx)
            if edge in self.noninterdictable_edges:
                self.noninterdictable_indices.append(idx)
        
        self.source_nodes = []
        for edge_id in self.edge_groups[self.super_source_nodes[0]]['in']:
            self.source_nodes.append(edge_id[0])
        self.sink_nodes = []
        for edge_id in self.edge_groups[self.super_sink_nodes[0]]['in']:
            self.sink_nodes.append(edge_id[0])
        self.num_sink_nodes = len(self.sink_nodes)

        # Pre-compute reverse edges for caching speed
        self.reverse_edges_list = [(e[1], e[0]) for e in self.both_edges]
            
    def _setup_spaces(self):
        """Setup observation and action spaces based on environment configuration."""
        # Calculate space dimensions
        self.num_both_edges = np.int64(len(self.both_edges))
        
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
    
    def solve_max_flow(self, capacity_dict=None, routing_assumption='zero_sum'):
        """
        Solve the Max Flow network problem with strategy-specific objectives.
    
        Parameters:
        -----------
        capacity_dict : dict, optional
            If provided, uses this capacity dictionary instead of current state.
            Useful for batch solving with different capacity configurations.
        routing_assumption : 'zero_sum', 'isolate', 'canalize', 'divert'
            Determines secondary routing of defender over the network to thwart attacker's strategy
    
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

        # Update objectives if needed (e.g. start of new episode or model re-init)
        if (not getattr(self, 'strategy_objectives_setup', False)) or (self.old_routing_assumption!=routing_assumption):
            self._set_strategy_objectives(routing_assumption)
            self.strategy_objectives_setup = True
            self.old_routing_assumption = routing_assumption
            
        # Solve and return results
        self.maxflow_model.params.Seed = 1
        self.maxflow_model.optimize()
        
        if self.maxflow_model.Status == grb.GRB.OPTIMAL:
            obj_val = round(self.maxflow_model.ObjVal)
            flow_results = {e: round(var.X) for e, var in self.flow_var.items()}
        else:
            obj_val = 0
            flow_results = {e: 0 for e in self.flow_var.keys()}
        
        return obj_val, flow_results 

    def _initialize_maxflow_model(self):
        """Initialize the Gurobi max flow model with variables and constraints."""
        self.maxflow_model = grb.Model("Max Flow", env=self.GUROBI_ENV)
        self.super_edge = (self.max_num_nodes, 1)
        
        # Prepare edge list with super sink-source connection
        self.mf_all_both_edges = self.all_both_edges + [self.super_edge] 

        ##VARIABLES
        # Add Flow variables
        self.flow_var = self.maxflow_model.addVars(self.mf_all_both_edges, vtype=grb.GRB.CONTINUOUS, lb=0, name="flow_var")

        # Add Edge Usage variables
        # CHANGE: vtype=grb.GRB.BINARY -> vtype=grb.GRB.CONTINUOUS for speed
        self.edge_used = self.maxflow_model.addVars(self.all_both_edges, vtype=grb.GRB.CONTINUOUS, name="edge_used")

        ##CONSTRAINTS
        # Flow conservation for intermediate nodes
        self.maxflow_model.addConstrs(
            (grb.quicksum(self.flow_var[e] for e in self.edge_groups[n]['out']) == 
             grb.quicksum(self.flow_var[e] for e in self.edge_groups[n]['in'])
             for n in self.intermediate_nodes), name="flow_conservation"
        )
    
        # Super source and super sink flow conservation
        self.maxflow_model.addConstr(self.flow_var[self.super_edge] - grb.quicksum(self.flow_var[e] for e in self.edge_groups[1]['out']) + grb.quicksum(self.flow_var[e] for e in self.edge_groups[1]['in'])== 0,
                                     name="super_source_conservation"
        )
    
        self.maxflow_model.addConstr(grb.quicksum(self.flow_var[e] for e in self.edge_groups[self.super_sink_nodes[0]]['in']) -grb.quicksum(self.flow_var[e] for e in self.edge_groups[self.super_sink_nodes[0]]['out']) -
                                     self.flow_var[self.super_edge] == 0, name="super_sink_conservation"
        )
    
        # One-way flow constraints
        self.maxflow_model.addConstrs((self.edge_used[(u, v)] + self.edge_used[(v, u)] <= 1 
                                      for u, v in self.both_edges), name="one_way_flow"
        )
    
        # Minimum flow forward and reverse constraints
        self.maxflow_model.addConstrs((self.flow_var[e] >= self.edge_used[e] for e in self.all_both_edges), name="min_flow_forward")

        # Prevent backflow from Super Sink
        self.maxflow_model.addConstrs(
            (self.flow_var[e] == 0 
             for e in self.edge_groups[self.super_sink_nodes[0]]['out']),
            name="no_backflow_from_supersink"
        )

        # Prevent backflow to Super Source
        self.maxflow_model.addConstrs(
            (self.flow_var[e] == 0 
             for e in self.edge_groups[self.super_source_nodes[0]]['in']),
            name="no_backflow_to_supersource"
        )

    def _update_capacity_constraints(self):
        """LEGACY Update edge capacity constraints based on current interdiction state."""
        # Calculate current edge capacities considering interdiction
        # Ensure probabilities are float to avoid integer power errors
        probs = self.state["edge_interdiction_probability"][:self.num_both_edges].astype(float)
        interdicted = self.state["edge_interdicted"][:self.num_both_edges].astype(int)
    
        # Compute success probabilities
        success_probs = (1.0 - probs) ** interdicted
    
        # Generate binomial outcomes
        upper_bounds = np.random.binomial(1, success_probs) * self.state["edge_capacity"][:self.num_both_edges]
    
        # Remove old capacity constraints if they exist
        if hasattr(self, 'forward_cons'):
            self.maxflow_model.remove(self.forward_cons)
            self.maxflow_model.remove(self.reverse_cons)

        # Single batch addition for forward constraints
        self.forward_cons = self.maxflow_model.addConstrs((
            self.flow_var[e] <= upper_bounds[idx % self.num_both_edges] * self.edge_used[e] 
            for idx, e in enumerate(self.both_edges)), name="flow_capacity_forward")

        self.reverse_cons = self.maxflow_model.addConstrs((
            self.flow_var[(e[1],e[0])] <= upper_bounds[idx % self.num_both_edges] * self.edge_used[(e[1],e[0])] 
            for idx, e in enumerate(self.both_edges)), name="flow_capacity_reverse")

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
            self.maxflow_model.remove(self.reverse_cons)

        self.forward_cons = self.maxflow_model.addConstrs((
            self.flow_var[e] <= capacity_dict.get(e, 0) * self.edge_used[e] 
            for idx, e in enumerate(self.both_edges)), name="flow_capacity_forward")

        self.reverse_cons = self.maxflow_model.addConstrs((
            self.flow_var[(e[1],e[0])] <= capacity_dict.get(e, 0) * self.edge_used[(e[1],e[0])] 
            for idx, e in enumerate(self.both_edges)), name="flow_capacity_reverse")        
        
    def _set_strategy_objectives(self, routing_assumption):
        """Set hierarchical objectives based on attacker strategy."""
        # Clear existing objectives
        self.maxflow_model.NumObj = 0
        self.maxflow_model.ModelSense = grb.GRB.MAXIMIZE
        self.maxflow_model.update()
        
        # Clean up previous auxiliary variables/constraints
        if hasattr(self, 'aux_vars'):
            for v in self.aux_vars: 
                try: self.maxflow_model.remove(v)
                except: pass
            self.aux_vars = []
        if hasattr(self, 'aux_constrs'):
            for c in self.aux_constrs: 
                try: self.maxflow_model.remove(c)
                except: pass
            self.aux_constrs = []
        
        self.aux_vars = []
        self.aux_constrs = []
        
        # 1. Primary Objective: Maximize Total Flow (Always Priority 10)
        self.maxflow_model.setObjectiveN(self.flow_var[self.super_edge], index=0, priority=10, weight=1.0, name="max_flow")

        # Minimize number of edges used to prevent cycles (Always Priority 1)
        expr = grb.quicksum(self.edge_used[e] for e in self.all_both_edges)
        self.maxflow_model.setObjectiveN(expr, index=1, priority=1, weight=-1.0, name="min_edges_used")

        if routing_assumption == "zero_sum":
            pass

        elif routing_assumption == "isolate":
            # Secondary: Maximize flow to isolated edges (Priority 5)
            target_indices = np.where(self.state['isolate_objective'][:self.num_both_edges] == 1)[0]
            target_edges = [self.both_edges[i] for i in target_indices]
            
            if target_edges:
                expr = grb.quicksum(self.flow_var[e] + self.flow_var[(e[1], e[0])] for e in target_edges)
                self.maxflow_model.setObjectiveN(expr, index=1, priority=5, weight=1.0, name="max_isolate_flow")

        elif routing_assumption == "canalize":
            # Secondary: Maximize minimum flow along canalized edges (Priority 5)
            target_indices = np.where(self.state['canalize_objective'][:self.num_both_edges] == 1)[0]
            target_edges = [self.both_edges[i] for i in target_indices]
            
            if target_edges:
                # Aux variable z representing min flow
                z = self.maxflow_model.addVar(vtype=grb.GRB.CONTINUOUS, name="min_canalize_flow")
                self.aux_vars.append(z)
                
                # Constraints: z <= flow_e
                constrs = self.maxflow_model.addConstrs(
                    (z <= self.flow_var[e] + self.flow_var[(e[1], e[0])] for e in target_edges),
                    name="min_flow_constr"
                )
                self.aux_constrs.extend(constrs.values())
                
                self.maxflow_model.setObjectiveN(z, index=1, priority=5, weight=1.0, name="max_min_canalize")

        elif routing_assumption == "divert":
            # Secondary: Maximize min flow on 'divert_from' (Priority 5)
            # Tertiary: Minimize flow on 'divert_to' (Priority 3)
            
            # Divert From
            from_indices = np.where(self.state['divert_from_objective'][:self.num_both_edges] == 1)[0]
            from_edges = [self.both_edges[i] for i in from_indices]
            
            if from_edges:
                z_from = self.maxflow_model.addVar(vtype=grb.GRB.CONTINUOUS, name="min_divert_from_flow")
                self.aux_vars.append(z_from)
                
                constrs = self.maxflow_model.addConstrs(
                    (z_from <= self.flow_var[e] + self.flow_var[(e[1], e[0])] for e in from_edges),
                    name="min_from_constr"
                )
                self.aux_constrs.extend(constrs.values())
                
                self.maxflow_model.setObjectiveN(z_from, index=1, priority=5, weight=1.0, name="max_min_divert_from")
            
            # Divert To
            to_indices = np.where(self.state['divert_to_objective'][:self.num_both_edges] == 1)[0]
            # Only include edges in divert_to that are NOT in divert_from
            to_indices = np.setdiff1d(to_indices, from_indices)
            to_edges = [self.both_edges[i] for i in to_indices]
            
            if to_edges:
                expr_to = grb.quicksum(self.flow_var[e] + self.flow_var[(e[1], e[0])] for e in to_edges)
                # Minimize -> weight = -1.0
                self.maxflow_model.setObjectiveN(expr_to, index=2, priority=3, weight=-1.0, name="min_divert_to_sum")

        self.maxflow_model.update()
    
    # BEGIN Gymnasium Environment Methods        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state and return observation."""
        # Clean up any existing models
        self._cleanup_models()
        self.strategy_objectives_setup = False # Force objective reset on next solve
        self.old_routing_assumption = False

        # Clear local outcome cache on reset because capacities/objectives change
        self.local_outcome_cache = {}

        # Clear centralized outcome cache if it exists
        if self.outcome_memo_actors:
            for actor in self.outcome_memo_actors:
                actor.clear.remote()
        elif self.outcome_memo_actor:
            self.outcome_memo_actor.clear.remote()
        
        # Call parent reset and set random seeds
        super().reset(seed=seed)
        if seed is not None:
            self._set_random_seeds(seed)

        # Generate network parameters
        # Sample edge capacities
        raw_capacities = self.base_spaces['edge_capacity'].sample()[:self.num_both_edges]
        edge_capacities = ((raw_capacities / 100.0) * (self.DEFAULT_TRAINING_EDGE_CAPACITY_RANGE[1]-self.DEFAULT_TRAINING_EDGE_CAPACITY_RANGE[0]) + self.DEFAULT_TRAINING_EDGE_CAPACITY_RANGE[0]).astype(int)
        if self.MAX_SOURCE_FLOW is not None:
            edge_capacities[self.super_source_out_indices] = self.DEFAULT_TRAINING_EDGE_CAPACITY_RANGE[0] * np.random.uniform(0.5, self.MAX_SOURCE_FLOW)
        if self.MAX_SINK_NEED is not None:
            edge_capacities[self.super_sink_in_indices] = self.DEFAULT_TRAINING_EDGE_CAPACITY_RANGE[0] * np.random.uniform(0.5, self.MAX_SINK_NEED)
        
        # Sample edge costs and interdiction probabilities
        raw_costs = self.base_spaces['edge_costs'].sample()[:self.num_both_edges]
        edge_costs = (((raw_costs) / 10) * (self.DEFAULT_TRAINING_EDGE_COST_RANGE[1]-self.DEFAULT_TRAINING_EDGE_COST_RANGE[0]) + self.DEFAULT_TRAINING_EDGE_COST_RANGE[0]).astype(int)
        self.min_edge_cost = np.min(edge_costs)
        
        #Sample interdiction probabilities based on deterministic setting
        if self.deterministic_outcomes:
            edge_interdiction_probabilities = np.ones(self.num_both_edges, dtype=np.float32)
        else:
            probs = self.base_spaces['edge_interdiction_probability'].sample()[:self.num_both_edges]
            # Round to 0.25 increments for consistency
            sample_rounded = np.round(probs * 20) #Trying replacing 4 with 20 to reduce symmetrical answers
            edge_interdiction_probabilities = (sample_rounded.astype(float) / 20)
        edge_interdiction_probabilities[self.noninterdictable_indices]=0
    
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
            _, self.reference_flows = self.solve_max_flow(routing_assumption = 'divert')
            from_flow = self._calculate_target_path_flow(self.reference_flows, 'divert_from_objective')
            to_flow = self._calculate_target_path_flow(self.reference_flows, 'divert_to_objective')
            self.reference_start_flows = (from_flow, to_flow)
            self.reference_obj = 0
        
        self.last_obj = self.reference_obj
        self.reference_budget = remaining_budget[0]

        self._cache_flow_array()
        
        self.num_interdictable = min(self.num_both_edges, self.action_space.n)
        self.has_probability = self.state['edge_interdiction_probability'][:self.num_interdictable] > 0
        self.has_capacity = self.state['edge_capacity'][:self.num_interdictable] > 0
        
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
        for edge, cap, cost, prob in zip(self.both_edges,
                                         network_params['capacities'],
                                         network_params['costs'],
                                         network_params['probabilities']):
            e = self.edges_episode[edge]
            e.capacity = cap
            e.interdiction_cost = cost
            e.interdiction_probability = prob

        # Pad all edge arrays to max_num_edges
        padded_capacities = np.zeros(self.max_num_edges, dtype=int)
        padded_capacities[:self.num_both_edges] = network_params['capacities']
    
        padded_costs = np.zeros(self.max_num_edges, dtype=int)
        padded_costs[:self.num_both_edges] = network_params['costs']
    
        padded_probabilities = np.zeros(self.max_num_edges, dtype=np.float32)
        padded_probabilities[:self.num_both_edges] = network_params['probabilities']
        
        # Create node arrays
        departure_nodes = np.zeros(self.max_num_edges, dtype=int)
        arrival_nodes = np.zeros(self.max_num_edges, dtype=int)
        departure_nodes[:self.num_both_edges] = np.array(self.edge_departures)
        arrival_nodes[:self.num_both_edges] = np.array(self.edge_arrivals)

        # CREATE EXPLICIT PADDING MASK
        padding_mask = np.zeros(self.max_num_edges, dtype=np.float32)
        padding_mask[:self.num_both_edges] = 1.0  # Mark valid edges as 1
    
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
        edge_in_path = np.array([edge in path_edges or (edge[1], edge[0]) in path_edges for edge in self.both_edges], dtype=bool)

        # Vectorized assignment
        canalize_objective[:len(edge_in_path)] = edge_in_path.astype(int)

        return {**base_state, 'canalize_objective': canalize_objective}

    def _add_isolate_components(self, base_state):
        """Add isolate-specific objective to state (edge-based, sink-connected only)."""
        # Create padded objective with at least 1 marked sink edge
        isolate_objective = np.zeros(self.max_num_edges, dtype=int)
        num_to_mark = np.random.randint(1, self.num_sink_nodes + 1)
        chosen_nodes = np.random.choice(self.sink_nodes, size = num_to_mark, replace=False)
        
        # Mark edges where arrival is chosen_node AND departure is NOT super sink
        arrival_mask = np.isin(self.edge_arrivals, chosen_nodes)
        departure_mask = ~np.isin(self.edge_departures, self.super_sink_nodes)
        
        marked_indices = np.where(arrival_mask & departure_mask)[0].tolist()
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
        divert_to = np.zeros(self.max_num_edges, dtype=int)

        for e, edge in enumerate(self.both_edges):
            if edge in from_path or (edge[1],edge[0]) in from_path:
                divert_from[e] = 1
            if edge in to_path or (edge[1],edge[0]) in to_path:
                divert_to[e] = 1
             # Padded entries remain 0
        return {**base_state, 'divert_from_objective': divert_from, 'divert_to_objective': divert_to}

    def _find_simple_path(self):
        """Find a simple path from source to sink."""
        path_edges = []
        current_node = random.choice(self.source_nodes)
        visited = {1}
        sink = self.super_sink_nodes[0]
    
        while current_node not in self.sink_nodes:
            valid_edges = [e for e in self.edge_groups[current_node]['out'] if e[1] not in visited and e[1] >= current_node - 1]
        
            if not valid_edges or len(visited)>self.MAX_PATH_LENGTH:
                # Restart if stuck
                current_node = random.choice(self.source_nodes)
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
        sink = self.super_sink_nodes[0]
    
        while current_node != sink:
            outgoing_edges = self.edge_groups[current_node]['out']
            next_edge = max(outgoing_edges, key=lambda e: flows.get(e, 0)+ random.random() * 1e-6)
            from_path.add(next_edge)
            current_node = next_edge[1]
    
        return from_path

    def _find_alternative_path(self, max_flow_edges):
        """Find an alternative path avoiding specified edges."""
        adj = {}
        for u, v in max_flow_edges:
            adj[u] = v

        # Trace path from source (1) to sink (250)
        max_flow_path = []
        current = 1
        while current != 250:
            next_node = adj[current]
            max_flow_path.append((current, next_node))
            current = next_node
        
        # Track attempted breakpoints to avoid infinite loops
        attempted_breakpoints = set()
        max_attempts = len(max_flow_path[1:-2])
    
        while len(attempted_breakpoints) < max_attempts:
            # Choose a random breakpoint that hasn't been tried yet
            available_breakpoints = [bp for bp in max_flow_path[1:-2] if bp not in attempted_breakpoints]
        
            if not available_breakpoints:
                # If all breakpoints have been tried, fall back to original path
                return set(max_flow_path)
        
            breakpoint = random.choice(available_breakpoints)
            attempted_breakpoints.add(breakpoint)
        
            breakpoint_index = max_flow_path.index(breakpoint)
            path_to_keep = max_flow_path[:breakpoint_index+1]
            path_to_avoid = max_flow_path[breakpoint_index+1:]
        
            # Try to find alternative path from this breakpoint
            path_edges = set(path_to_keep)
            current_node = breakpoint[1]
            visited = {1} | {edge[1] for edge in path_to_keep}
            sink = self.super_sink_nodes[0]
        
            stuck_count = 0
            max_stuck_attempts = 5  # Limit retries before choosing new breakpoint
        
            while current_node != sink:
                valid_edges = []
                for edge in self.edge_groups[current_node]['out']:
                    target = edge[1]
                    if (target not in visited and target >= current_node - 1 and 
                        edge not in path_to_avoid):
                
                        # Check for valid future moves
                        if target != sink:
                            future_valid = any(
                                e not in path_to_avoid and e[1] != current_node and e[1] >= e[0] - 1
                                for e in self.edge_groups[target]['out']
                            )
                            if not future_valid:
                                continue
                
                        valid_edges.append(edge)
            
                if not valid_edges or len(visited) > self.MAX_PATH_LENGTH + len(path_to_keep):
                    stuck_count += 1
                    if stuck_count >= max_stuck_attempts:
                        # This breakpoint doesn't work, try a new one
                        break
                
                    # Try restarting from the breakpoint
                    current_node = breakpoint[1]
                    visited = {1} | {edge[1] for edge in path_to_keep}
                    path_edges = set(path_to_keep)
                    continue
        
                selected_edge = random.choice(valid_edges)
                path_edges.add(selected_edge)
                visited.add(selected_edge[1])
                current_node = selected_edge[1]
        
            # If we successfully reached the sink, return the alternative path
            if current_node == sink:
                return path_edges
    
        # If no alternative path found after trying all breakpoints, return original
        return set(max_flow_path)
    
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
        # Use mask_fn to validate action (unified logic)
        action_mask = self.mask_fn(depth=1)
        
        # Check if action is valid:
        # 1. Must be within actual edges (not padding)
        # 2. Must be allowed by mask_fn
        valid_action = (action < self.num_both_edges) and (action_mask[action] == 1)
        
        # Apply action effects
        if valid_action:
            # Deduct cost from budget
            remaining_budget[0] = remaining_budget[0] - self.state['edge_costs'][action]
    
            # Mark edge as interdicted
            self.state['edge_interdicted'][action] += 1
            
            #Compute Rewards
            strategy_calculators = {"zero_sum": self._calculate_zero_sum_reward,
                                    "canalize": self._calculate_canalize_reward,
                                    "isolate": self._calculate_isolate_reward,
                                    "divert": self._calculate_divert_reward}
            calculator = strategy_calculators.get(self.attacker_strategy)
            reward = calculator()
            self._cache_flow_array()
        else:
            #Determine penalty and decrement budget
            reward = self.PENALTY_VALUE
            remaining_budget[0] = max(0, remaining_budget[0] - self.state['edge_costs'][action])

        # Check if episode is complete
        done = self._is_episode_complete(remaining_budget)
    
        # Update state
        self.state['budget'] = remaining_budget
    
        return self.state, float(reward), bool(done), False, {}
        
    def _calculate_zero_sum_reward(self):                         
        """Calculate reward for zero-sum strategy (maximize disruption)."""
        objective_value, self.reference_flows = self._compute_objective_and_flows()
        reward = max(self.last_obj - objective_value, 0) / self.reference_budget
        if reward > 0:
            self.last_obj = objective_value   
        elif reward == 0:
            reward = self.PENALTY_VALUE
        return reward

    def _calculate_stochastic_objective_and_flow(self, strategy_type="zero_sum", return_full_flows=False):
        """
        Optimized stochastic calculation: group by unique outcomes and weight by probability.
    
        This method:
        1. Samples interdiction outcomes (success/failure) based on probabilities
        2. Groups identical outcomes together
        3. Solves max flow once per unique outcome
        4. Computes weighted average based on outcome frequencies
        """
        # Extract interdiction probabilities
        probs = self.state['edge_interdiction_probability'][:self.num_both_edges]
        interdicted = self.state['edge_interdicted'][:self.num_both_edges].astype(int)
        total_samples = self.SAMPLE_SIZE
        
        unique_outcomes = []
        outcome_weights = {}

        if total_samples is None:
            # Systematic approach: Generate all possible outcomes
            interdicted_indices = np.where(interdicted > 0)[0]
            num_interdicted = len(interdicted_indices)
            
            # Calculate success probabilities for interdicted edges
            # P(success) = 1 - (1 - p)^k
            edge_success_probs = 1 - (1 - probs[interdicted_indices]) ** interdicted[interdicted_indices]
            
            for outcome_combo in product([0, 1], repeat=num_interdicted):
                outcome_array = np.zeros(self.num_both_edges, dtype=int)
                prob = 1.0
                
                for i, idx in enumerate(interdicted_indices):
                    success = outcome_combo[i]
                    outcome_array[idx] = success
                    
                    if success == 1:
                        prob *= edge_success_probs[i]
                    else:
                        prob *= (1 - edge_success_probs[i])
                
                outcome = tuple(outcome_array)
                unique_outcomes.append(outcome)
                outcome_weights[outcome] = prob
        else:
            # Sample interdiction outcomes
            outcome_samples = []
            for _ in range(total_samples):
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
            outcome_weights = {outcome: count / total_samples for outcome, count in outcome_counts.items()}

        # --- MEMOIZATION START ---
        outcomes_needed_from_central = []
        
        # 1. Check Local Cache
        for outcome in unique_outcomes:
            # Check if outcome is in cache AND meets data requirements (ie flows vs no flows)
            is_valid_hit = outcome in self.local_outcome_cache
            if is_valid_hit and return_full_flows and 'flows' not in self.local_outcome_cache[outcome]:
                is_valid_hit = False
            
            if not is_valid_hit:
                outcomes_needed_from_central.append(outcome)
        
        # 2. Check Central Cache (only for what wasn't in local)
        outcomes_to_solve = []
        if outcomes_needed_from_central and self.outcome_memo_actors:
            import zlib
            num_shards = len(self.outcome_memo_actors)
            
            # Group keys by shard
            shard_keys = defaultdict(list)
            for outcome in outcomes_needed_from_central:
                # Use stable hash
                shard_idx = zlib.adler32(str(outcome).encode()) % num_shards
                shard_keys[shard_idx].append(outcome)
            
            # Batch fetch from Ray (parallel)
            futures = []
            shard_indices = []
            for idx, keys in shard_keys.items():
                futures.append(self.outcome_memo_actors[idx].get_batch.remote(keys))
                shard_indices.append(idx)
            
            if futures:
                all_results = ray.get(futures)
                
                # Process results
                for i, results in enumerate(all_results):
                    keys = shard_keys[shard_indices[i]]
                    for outcome, res in zip(keys, results):
                        # Check if remote result is valid
                        is_valid_result = res is not None
                        if is_valid_result and return_full_flows and 'flows' not in res:
                            is_valid_result = False
                            
                        if is_valid_result:
                            self.local_outcome_cache[outcome] = res #(outcome, strategy_type)
                        else:
                            outcomes_to_solve.append(outcome)
        else:
            outcomes_to_solve = outcomes_needed_from_central
        # --- MEMOIZATION END ---

        # 3. Solve Max Flow for truly missing outcomes
        new_results_for_central = {}
        for outcome in outcomes_to_solve:
            # Convert outcome to capacity dict
            capacity_dict = {}
            for idx, edge in enumerate(self.both_edges):
                base_capacity = self.state['edge_capacity'][idx]
                is_interdicted = outcome[idx]
                capacity_dict[edge] = 0 if is_interdicted else base_capacity
        
            # Solve max flow
            obj, flows = self.solve_max_flow(capacity_dict, routing_assumption=strategy_type)
        
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
        
            res = {
                'objective': objective
            }
            
            if return_full_flows:
                res['flows'] = flows
            else:
                # Compression logic
                if self.state['budget'][0] < self.min_edge_cost:
                    res['nonzero_flow_indices'] = []
                else:
                    # Store indices where flow > 0
                    indices = []
                    for edge, flow in flows.items():
                        if flow > 0:
                            if edge in self.edge_to_index:
                                indices.append(self.edge_to_index[edge])
                            elif (edge[1], edge[0]) in self.edge_to_index:
                                indices.append(self.edge_to_index[(edge[1], edge[0])])
                    res['nonzero_flow_indices'] = list(set(indices))
            
            # Update local cache
            self.local_outcome_cache[outcome] = res #(outcome, strategy_type)
            # Queue for central update
            new_results_for_central[outcome] = res
            
        # 4. Update Central Cache (Async / Fire-and-forget)
        if new_results_for_central and self.outcome_memo_actors:
            import zlib
            num_shards = len(self.outcome_memo_actors)
            
            shard_updates = defaultdict(lambda: ([], [])) # (keys, values)
            
            for outcome, res in new_results_for_central.items():
                shard_idx = zlib.adler32(str(outcome).encode()) % num_shards
                keys, values = shard_updates[shard_idx]
                keys.append(outcome)
                values.append(res)
            
            for idx, (keys, vals) in shard_updates.items():
                self.outcome_memo_actors[idx].set_batch.remote(keys, vals)

        # 5. Compute weighted averages using Local Cache (which is now fully populated)
        weighted_objective = 0.0
        weighted_flows = defaultdict(float)
        
        for outcome in unique_outcomes:
            result = self.local_outcome_cache[outcome]  #(outcome, strategy_type)
            weight = outcome_weights[outcome]
            
            weighted_objective += result['objective'] * weight
            
            if 'nonzero_flow_indices' in result:
                # Reconstruct flow as 1 for these indices
                for idx in result['nonzero_flow_indices']:
                    edge = self.both_edges[idx]
                    weighted_flows[edge] += 1.0 * weight
            elif 'flows' in result:
                for edge, flow in result['flows'].items():
                    weighted_flows[edge] += flow * weight
    
        return weighted_objective, dict(weighted_flows)
    
    def _compute_objective_and_flows(self, deterministic_mode=None):
        """Calculate the max flow objective and edge flows."""
        if deterministic_mode is None:
            deterministic_mode = self.deterministic_outcomes
        
        if deterministic_mode:
            objective, flows = self.solve_max_flow()
        else:
            # Stochastic outcome calculation
            objective, flows = self._calculate_stochastic_objective_and_flow('zero_sum')
    
        return objective, flows

    def _calculate_canalize_objective_and_flows(self):
        """Calculate objective for canalize strategy (flow through specific path)."""
        if self.deterministic_outcomes:
            _, flows = self.solve_max_flow(routing_assumption = 'canalize')
            target_path_flow = self._calculate_target_path_flow(flows, 'canalize_objective')
            return target_path_flow, flows
        else:
            # Stochastic calculation - returns mean objective directly
            objective, mean_flows = self._calculate_stochastic_objective_and_flow('canalize')
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
            _, flows = self.solve_max_flow(routing_assumption = 'isolate')
            target_node_flow = self._calculate_target_edge_flow(flows, 'isolate_objective')
            return target_node_flow, flows
        else:
            # Stochastic calculation - returns mean objective directly
            objective, mean_flows = self._calculate_stochastic_objective_and_flow('isolate')
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
            _, flows = self.solve_max_flow(routing_assumption = 'divert')
            from_flow = self._calculate_target_path_flow(flows, 'divert_from_objective')
            to_flow = self._calculate_target_path_flow(flows, 'divert_to_objective')
            diverted_flow_from = self.reference_start_flows[0] - from_flow
            diverted_flow_to = to_flow - self.reference_start_flows[1] 
            objective = np.min([diverted_flow_from,diverted_flow_to])
            
            return objective, flows
        else:
            # Stochastic calculation - returns mean objectives directly
            mean_objective, mean_flows = self._calculate_stochastic_objective_and_flow('divert')
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

        current_node = 1
        visited = {1}

        while current_node not in self.sink_nodes and current_node not in self.super_sink_nodes:
            found_next = False
            if current_node in self.edge_groups:
                for edge in self.edge_groups[current_node]['out']:
                    neighbor = edge[1]
                    
                    if neighbor in visited:
                        continue
                        
                    idx = self.edge_to_index.get(edge)
                    if idx is None:
                        idx = self.edge_to_index.get((neighbor, current_node))
                        
                    if idx is not None and objective[idx] == 1:
                        target_flows.append(flows.get(edge, 0))
                        current_node = neighbor
                        visited.add(current_node)
                        found_next = True
                        break
            
            if not found_next:
                break

        if not target_flows:
            return 0.0

        # Return minimum flow among target edges
        return min(target_flows)

    def _calculate_target_edge_flow(self, flows, objective_key):
        """Calculate total flow on edges marked in the objective."""
        objective = self.state[objective_key]
        # Get indices where objective is 1
        target_indices = np.where(objective[:self.num_both_edges] == 1)[0]
        
        target_edges = [self.both_edges[i] for i in target_indices]

        # Filter to only include edges outgoing to super sink nodes to avoid double counting
        # and internal flow between sink nodes.
        #if objective_key == 'isolate_objective':
        #     target_edges = [e for e in target_edges if e[1] in self.super_sink_nodes]
    
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

    def render(self, mode='human', indices=25):
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
    
        print(f"Actual number of edges in graph: {self.num_both_edges}")
    
        # Budget information
        print(f"\n{'Budget Information':^80}")
        print("-" * 80)
        print(f"Remaining / Reference Budget: {self.state['budget'][0]} / {self.reference_budget}")
    
        # Edge-based state information (filtered by padding mask)
        print(f"\n{'Edge State Information':^80}")
        print("-" * 80)
        print(f"{'Index':<8} {'Origin':<8} {'Dest':<8} {'Capacity':<12} {'Cost':<8} {'Int. Prob':<12} {'Interdicted':<12}")
        print("-" * 80)
    
        for idx in valid_indices[:indices]:  # Show first 25 valid edges
            capacity = self.state['edge_capacity'][idx]
            cost = self.state['edge_costs'][idx]
            prob = self.state['edge_interdiction_probability'][idx]
            interdicted = self.state['edge_interdicted'][idx]
            departure_node = self.state['edge_departure_node'][idx]
            arrival_node = self.state['edge_arrival_node'][idx]
        
            print(f"{idx:<8} {departure_node:<8} {arrival_node:<8} {capacity:<12} {cost:<8} {prob:<12.2f} {interdicted:<12}")
    
        if len(valid_indices) > indices:
            print(f"... ({len(valid_indices) - indices} more valid edges)")
    
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

                    # Print the actual edges with value 1
                    target_edge_indices = valid_indices[objective_values == 1]
                    target_edges = [self.both_edges[idx] for idx in target_edge_indices]
                    print(f"  Target edges: {target_edges}")
    
        # Additional state fields (non-edge specific)
        print(f"\n{'Other State Information':^80}")
        print("-" * 80)
    
        print(f"Last Objective / Reference Objective: {self.last_obj} / {self.reference_obj}")

        print("=" * 80)
        # END Gymnasium Environment Methods
            
    def solve_optimal_interdiction(self):
        if self.deterministic_outcomes == True: #Solve Deterministic Case with Wood's Max/Min Formulation
            if not hasattr(self, 'optimal_deterministic_model'):
                # Initialize the Gurobi model
                self.optimal_deterministic_model = grb.Model("Network Interdiction Model 1U", env=self.GUROBI_ENV)
                
                # Define Decision Variables
                self.alpha = self.optimal_deterministic_model.addVars(self.nodes.keys(), vtype=grb.GRB.BINARY, name="alpha")
                self.beta = self.optimal_deterministic_model.addVars(self.both_edges, vtype=grb.GRB.BINARY, name="beta")
                self.gamma = self.optimal_deterministic_model.addVars(self.both_edges, vtype=grb.GRB.BINARY, name="gamma")
                
                # Define Constraints
                self.optimal_deterministic_model.addConstrs(
                    (self.alpha[e[0]] - self.alpha[e[1]] + self.beta[e] + self.gamma[e] >= 0 for e in self.both_edges),
                    name="flow_conservation")

                self.optimal_deterministic_model.addConstrs(
                    (self.alpha[e[1]] - self.alpha[e[0]] + self.beta[e] + self.gamma[e] >= 0 for e in self.both_edges),
                    name="flow_conservation_reverse")

                self.optimal_deterministic_model.addConstr(
                    self.alpha[self.super_sink_nodes[0]] - self.alpha[self.super_source_nodes[0]] >= 1, name = "sink-source")
            
            # Update Constraints
            if hasattr(self, 'budget_constr'):
                self.optimal_deterministic_model.remove(self.budget_constr)

            self.budget_constr = self.optimal_deterministic_model.addConstr(
                grb.quicksum(self.edges_episode[e].interdiction_cost * self.gamma[e] for e in self.both_edges) <= self.state['budget'][0],
                name="budget")

            if hasattr(self, 'interdiction_success_constr'):
                self.optimal_deterministic_model.remove(self.interdiction_success_constr)

            self.interdiction_success_constr = self.optimal_deterministic_model.addConstrs(
                (self.gamma[e] <= self.edges_episode[e].interdiction_probability for e in self.both_edges),
                name="interdiction_success_upper_bound")

            # Define Objective Value
            self.optimal_deterministic_model.setObjective(
                grb.quicksum(edge.capacity * self.beta[edge_id] for edge_id, edge in self.edges_episode.items()), grb.GRB.MINIMIZE)

            # Optimize
            self.optimal_deterministic_model.optimize()

            interdicted_edges = [
                e for e in self.both_edges 
                if self.gamma[e].X > 0.99  # Account for floating point precision
            ]

            return self.optimal_deterministic_model.ObjVal, interdicted_edges
        
        else:  #Solve Stochastic Case with Cormican's Formulation      
            M = 100                       # Number of training episodes
            N = 700                   # Number of test episodes
            seed_list = [100, 200, 300]#, 400, 500]
            best_objective_value = 100000    # Big M Value
            best_interdicted_edges = []
            unique_interdicted_sets = []

            # Test multiple solutions
            for seed in seed_list:
                if self.multiple_interdiction_attempts:
                    objective_value, interdicted_edges, interdicted_quantities = self.solve_stochastic_max_flow_IM(n_scenarios=M, seed=seed)
                    
                    # Create dense vector for key (values > 1 allowed)
                    interdiction_vector = np.zeros(len(self.both_edges), dtype=int)
                    for edge, qty in zip(interdicted_edges, interdicted_quantities):
                        if edge in self.edge_to_index:
                            interdiction_vector[self.edge_to_index[edge]] = qty
                    interdicted_key = tuple(interdiction_vector)
                    
                else:
                    objective_value, interdicted_edges = self.solve_stochastic_max_flow(n_scenarios=M, seed=seed)
                    
                    # Create dense vector for key (binary)
                    interdiction_vector = np.zeros(len(self.both_edges), dtype=int)
                    for edge in interdicted_edges:
                        if edge in self.edge_to_index:
                            interdiction_vector[self.edge_to_index[edge]] = 1
                    interdicted_key = tuple(interdiction_vector)
                    
                # Check if the set of interdicted edges is unique
                if interdicted_key not in unique_interdicted_sets:
                    unique_interdicted_sets.append(interdicted_key)       

                    if self.multiple_interdiction_attempts:
                        objective_value, interdicted_edges, interdicted_quantities = self.solve_stochastic_max_flow_IM(n_scenarios=N, interdicted_edges=interdicted_edges, interdicted_quantities=interdicted_quantities)
                        # Expand for return value
                        current_solution = []
                        for e, k in zip(interdicted_edges, interdicted_quantities):
                            current_solution.extend([e] * k)
                    else:
                        objective_value, interdicted_edges = self.solve_stochastic_max_flow(n_scenarios=N, interdicted_edges=interdicted_edges)
                        current_solution = interdicted_edges

                    if objective_value < best_objective_value:
                        best_objective_value = objective_value
                        best_interdicted_edges = current_solution

            return best_objective_value, best_interdicted_edges

    def solve_stochastic_max_flow(self, n_scenarios = 50, seed = 173, interdicted_edges = []):      
        # Optimally Solve for Stochastic Solution using Model 1U and SAA
        if not hasattr(self, 'optimal_stochastic_model'):
            # Initializing the model
            self.optimal_stochastic_model = grb.Model("Stochastic Model", env=self.GUROBI_ENV)

            # Creating decision variables
            self.stochastic_gamma = self.optimal_stochastic_model.addVars(self.both_edges, vtype=grb.GRB.BINARY, name="gamma")

            # Create Variable Lower and Upper Bounds
            self.optimal_stochastic_model.setAttr("LB", [self.stochastic_gamma[e] for e in interdicted_edges],1)
            self.optimal_stochastic_model.setAttr("UB", [self.stochastic_gamma[e] for e in self.noninterdictable_edges],0)
            
             # Budget constraint
            self.stochastic_budget_constr = self.optimal_stochastic_model.addConstr(
                grb.quicksum(self.edges_episode[e].interdiction_cost * self.stochastic_gamma[e]
                             for e in self.both_edges) <= self.state['budget'][0], name="budget")

            self.stochastic_old_state = self.state
            self.stochastic_old_interdicted_edges = interdicted_edges

        if self.stochastic_old_interdicted_edges != interdicted_edges:
            # Update Variable Lower Bounds
            self.optimal_stochastic_model.setAttr("LB", [self.stochastic_gamma[e] for e in self.both_edges],0)
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
            self.stochastic_beta = self.optimal_stochastic_model.addVars([(e, s) for s in self.scenarios for e in self.both_edges],
                                                                          vtype=grb.GRB.BINARY, name="beta")

            if hasattr(self, 'stochastic_source_sink_constr'):
                self.optimal_stochastic_model.remove(self.stochastic_source_sink_constr)
                del self.stochastic_source_sink_constr 

            self.stochastic_source_sink_constr = self.optimal_stochastic_model.addConstrs(
                (self.stochastic_alpha[self.super_sink_nodes[0],s] - self.stochastic_alpha[self.super_source_nodes[0], s] >= 1
                for s in self.scenarios), name="source_sink")

            # Objective Function
            self.optimal_stochastic_model.setObjective((1/n_scenarios)*grb.quicksum(edge.capacity * self.stochastic_beta[edge_id, s]
                for s in self.scenarios for edge_id, edge in self.edges_episode.items()), grb.GRB.MINIMIZE)
        
        # Scenario generation
        scenario_outcomes = np.random.binomial(1, self.state["edge_interdiction_probability"][:self.num_both_edges], 
                                               size=(n_scenarios, len(self.both_edges))) #Generate a 1 for success and a 0 for failure
        
        if hasattr(self, 'stochastic_aabg_constr'):
            self.optimal_stochastic_model.remove(self.stochastic_aabg_constr)
            self.optimal_stochastic_model.remove(self.stochastic_aabg_reverse_constr)
            self.optimal_stochastic_model.update()  # Force model synchronization
            del self.stochastic_aabg_constr, self.stochastic_aabg_reverse_constr
            
        self.stochastic_aabg_constr = self.optimal_stochastic_model.addConstrs(
            (self.stochastic_alpha[e[0],s] - self.stochastic_alpha[e[1], s] + self.stochastic_beta[e, s] + (self.stochastic_gamma[e] * scenario_outcomes[s, edge_id]) >= 0 for s in self.scenarios for edge_id, e in enumerate(self.both_edges)), name='aabg')
        self.stochastic_aabg_reverse_constr = self.optimal_stochastic_model.addConstrs(
            (self.stochastic_alpha[e[1],s] - self.stochastic_alpha[e[0], s] + self.stochastic_beta[e, s] + (self.stochastic_gamma[e] * scenario_outcomes[s, edge_id]) >= 0 for s in self.scenarios for edge_id, e in enumerate(self.both_edges)), name='aabg')

        # Solving
        self.optimal_stochastic_model.optimize()

        interdicted_edges = [
            e for e in self.both_edges
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
            gamma_indices = [(e, k) for e in self.interdictable_edges for k in range(1, self.max_interdictions + 1)]
            self.stochastic_gamma_IM = self.optimal_stochastic_model_IM.addVars(gamma_indices, vtype=grb.GRB.BINARY, name="g_IM")
            self.optimal_stochastic_model_IM.update()

            # Create Variable Lower Bounds
            self.optimal_stochastic_model_IM.setAttr("LB", [self.stochastic_gamma_IM[e,k] for e, k in zip(interdicted_edges, interdicted_quantities)],1)

            # Gamma constraint
            self.stochastic_gamma_constr_IM = self.optimal_stochastic_model_IM.addConstrs((grb.quicksum(
                self.stochastic_gamma_IM[e,k] for k in range(1, self.max_interdictions + 1)) <= 1 for e in self.interdictable_edges), name="gamma_constr_IM")
            
             # Budget constraint
            self.stochastic_budget_constr_IM = self.optimal_stochastic_model_IM.addConstr(grb.quicksum(
                self.edges_episode[e].interdiction_cost * k * self.stochastic_gamma_IM[e,k] 
                for e in self.interdictable_edges for k in range(1, self.max_interdictions + 1)) <= self.state['budget'][0], name="budget_IM")

            self.stochastic_old_state_IM = self.state
            self.stochastic_old_interdicted_edges_IM = interdicted_edges
            self.stochastic_old_interdicted_quantities_IM = interdicted_quantities

        if self.stochastic_old_interdicted_edges_IM != interdicted_edges or self.stochastic_old_interdicted_quantities_IM != interdicted_quantities:
            # Update Variable Lower Bounds
            self.optimal_stochastic_model_IM.setAttr("LB", [self.stochastic_gamma_IM[e,k] for e in self.interdictable_edges for k in range(1, self.max_interdictions + 1)],0)
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

            self.stochastic_source_sink_constr_IM = self.optimal_stochastic_model_IM.addConstrs((self.stochastic_alpha_IM[self.super_sink_nodes[0],s] - self.stochastic_alpha_IM[self.super_source_nodes[0], s] >= 1 for s in self.scenarios_IM), name="source_sink_IM")

            # Objective Function
            self.optimal_stochastic_model_IM.setObjective((1/n_scenarios)*grb.quicksum(self.edges_episode[e].capacity * self.stochastic_beta_IM[e, s]
                for s in self.scenarios_IM
                for e in self.edges_reset), grb.GRB.MINIMIZE)
        
        # Scenario generation 
        # Compute base probability and ensure it's an array
        interdictable_indices = [self.edge_to_index[e] for e in self.interdictable_edges]
        p_base = self.state["edge_interdiction_probability"][interdictable_indices]

        # Create k values (1 to self.max_interdictions)
        k_vals = np.arange(1, self.max_interdictions + 1)

        # Calculate success probabilities: 1 - (1-p)^k for each edge and k
        probs = 1 - (1 - p_base[:, np.newaxis]) ** k_vals

        # Generate scenario outcomes
        scenario_outcomes = np.random.binomial(1, probs, size=(n_scenarios, len(self.interdictable_edges), len(k_vals)))
        
        interdictable_edge_map = {e: i for i, e in enumerate(self.interdictable_edges)}

        if hasattr(self, 'stochastic_aabg_constr_IM'):
            self.optimal_stochastic_model_IM.remove(self.stochastic_aabg_constr_IM)
            self.optimal_stochastic_model_IM.remove(self.stochastic_aabg_reverse_constr_IM)

            self.optimal_stochastic_model_IM.update()  # Force model synchronization
            del self.stochastic_aabg_constr_IM, self.stochastic_aabg_reverse_constr_IM
            
        self.stochastic_aabg_constr_IM = self.optimal_stochastic_model_IM.addConstrs((self.stochastic_alpha_IM[e[0],s] - self.stochastic_alpha_IM[e[1], s]+self.stochastic_beta_IM[e, s]+ (grb.quicksum(self.stochastic_gamma_IM[e,k] * scenario_outcomes[s, interdictable_edge_map[e], k-1] for k in k_vals) if e in interdictable_edge_map else 0) >= 0 for s in self.scenarios_IM for e in self.edges_reset.keys()), name='aabg_IM')
        self.stochastic_aabg_reverse_constr_IM = self.optimal_stochastic_model_IM.addConstrs((self.stochastic_alpha_IM[e[1],s] - self.stochastic_alpha_IM[e[0], s]+self.stochastic_beta_IM[e, s]+ (grb.quicksum(self.stochastic_gamma_IM[e,k] * scenario_outcomes[s, interdictable_edge_map[e], k-1] for k in k_vals) if e in interdictable_edge_map else 0) >= 0 for s in self.scenarios_IM for e in self.edges_reset.keys()), name='aabg_IM')

        # Solving
        self.optimal_stochastic_model_IM.optimize()

        # Extract interdiction decisions with k-values
        interdiction_decisions = []
        for e in self.interdictable_edges:
            for k in range(1, self.max_interdictions + 1):
                if self.stochastic_gamma_IM[e, k].X > 0.5:
                    interdiction_decisions.append((e, k))
                    break  # Only one k per edge possible

        # Extract just the edge list if needed
        interdicted_edges = [e for e, k in interdiction_decisions]
        interdicted_quantities = [k for e, k in interdiction_decisions]

        return (self.optimal_stochastic_model_IM.objVal, interdicted_edges, interdicted_quantities)

    def solve_backward_induction_OLD(self, verbose=False):  #REMOVE _OLD
        """
        Solve the optimal interdiction strategy for attacker using backward induction.
        This method finds the optimal interdictions for a particular attacker strategy.
    
        Returns:
            tuple: (optimal_objective_value, optimal_interdiction_sequence)
        """
        # Calculate minimum edge cost for depth estimation (only from real edges)
        real_edge_costs = self.state['edge_costs'][:self.num_both_edges]
        self.min_edge_cost = min(real_edge_costs[real_edge_costs > 0], default=float('inf'))
        if self.min_edge_cost == float('inf'):
            self.min_edge_cost = 1  # Fallback
    
        # Estimate total states
        max_budget = self.state['budget'][0]
        budget_levels = max_budget // self.min_edge_cost
        estimated_states = self.num_both_edges ** budget_levels  # Use actual_num_edges, not max
        update_rate = max(200, 1)
        memo = {}

        states_processed = 0
        pbar = tqdm(total=estimated_states, desc="DP States", unit=" states", disable=not verbose)

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
            # Initialize the dynamic programming table
            state_key = interdicted_state[:self.num_both_edges].tobytes()
        
            # Check if we've already solved this state
            if state_key in memo:
                update_progress(self.num_both_edges**(budget_levels-depth))
                return memo[state_key]

            old_budget = self.state['budget'][0]
            old_interdicted = self.state['edge_interdicted'].copy()  # Only copy this array
    
            self.state['budget'][0] = remaining_budget
            self.state['edge_interdicted'][:] = interdicted_state            
            
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
                
            #self.state = old_state
            self.state['budget'][0] = old_budget
            self.state['edge_interdicted'][:] = old_interdicted
            
            # Base case: no more budget or maximum depth reached
            if remaining_budget < self.min_edge_cost or depth >= 20:
                               
                memo[state_key] = (final_objective, [])
                update_progress(self.num_both_edges**(budget_levels-depth))
                return final_objective, []
       
            # Find all valid actions from current state
            action_mask = self.mask_fn()
            valid_actions = np.where(action_mask[:self.num_both_edges] == 1)[0]
            num_invalid_actions = np.where(action_mask[:self.num_both_edges] == 0)[0].shape[0]
            update_progress((self.num_both_edges**(budget_levels-(depth+1)))*num_invalid_actions)
        
            # If no valid actions, evaluate terminal state
            if len(valid_actions)==0:
                memo[state_key] = (final_objective, [])
                update_progress(self.num_both_edges**(budget_levels-depth))
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
        
        optimal_actions = [self.both_edges[idx] for idx in optimal_sequence]

        return optimal_reward, optimal_actions

    def load_network_from_state(self, seed, state):
        """Reset the environment to initial state and return observation."""
        # Clean up any existing models
        self._cleanup_models()
        self.strategy_objectives_setup = False # Force objective reset on next solve
        self.old_routing_assumption = False

        # Clear local outcome cache when loading new state
        self.local_outcome_cache = {}

        # Clear centralized outcome cache if it exists
        if self.outcome_memo_actors:
            for actor in self.outcome_memo_actors:
                actor.clear.remote()
        elif self.outcome_memo_actor:
            self.outcome_memo_actor.clear.remote()
        
        super().reset(seed=seed)
        if seed is not None:
            self._set_random_seeds(seed)
            
        network_params = {
            'capacities': state['edge_capacity'][:self.num_both_edges], 
            'costs': state['edge_costs'][:self.num_both_edges],          
            'probabilities': state['edge_interdiction_probability'][:self.num_both_edges],
            'budget': state['budget']
        }
        
        self.min_edge_cost = np.min(network_params['costs'])

        # Create base state
        base_state = self._create_base_state(network_params)

        self.state = state

        # Calculate reference objective value for the attacker's strategy
        if self.attacker_strategy == 'zero_sum':
            self.reference_obj, self.reference_flows = self._compute_objective_and_flows()
        elif self.attacker_strategy == 'canalize':
            self.reference_obj, self.reference_flows = self._calculate_canalize_objective_and_flows()
        elif self.attacker_strategy == 'isolate':
            self.reference_obj, self.reference_flows = self._calculate_isolate_objective_and_flows()
        elif self.attacker_strategy == 'divert':
            _, self.reference_flows = self.solve_max_flow(routing_assumption = 'divert')
            from_flow = self._calculate_target_path_flow(self.reference_flows, 'divert_from_objective')
            to_flow = self._calculate_target_path_flow(self.reference_flows, 'divert_to_objective')
            self.reference_start_flows = (from_flow, to_flow)
            self.reference_obj = 0
        
        self.last_obj = self.reference_obj
        self.reference_budget = state['budget'][0]

        self._cache_flow_array()

        self.num_interdictable = min(self.num_both_edges, self.action_space.n)
        self.has_probability = self.state['edge_interdiction_probability'][:self.num_interdictable] > 0
        self.has_capacity = self.state['edge_capacity'][:self.num_interdictable] > 0

        return self.state, {}

    def get_edges_on_paths_to_source(self, start_nodes=None):
        """
        Determine which edges have non-zero flow in self.cached_flow_array and lie along a path from the isolate_objective nodes to a source node.
        Uses backward BFS from target nodes through edges with non-zero flow to reach source nodes.
        Returns:
            np.ndarray: Boolean array of shape (self.numbothedges,) where True indicates the edge has non-zero flow and is on a path from an isolate_objective node to a source.
        """
        if start_nodes is None:
            start_nodes = self.state['isolate_objective']
            
        target_nodes = set()
        for idx in np.where(start_nodes[:self.num_both_edges]==1)[0]:
            edge = self.both_edges[idx]
            target_nodes.add(edge[1])

        visited_nodes = set(target_nodes)
        incoming_edge_indices = set()

        while target_nodes:
            arrival_in_targets = np.isin(self.edge_arrivals, list(target_nodes))
            has_flow = (self.cached_flow_array[:self.num_both_edges]) > 1e-6
            valid_edge_indices = np.where(arrival_in_targets & has_flow)[0]
            incoming_edge_indices.update(valid_edge_indices)
            # Get new target nodes (departure nodes of incoming edges)
            new_target_nodes = set([self.both_edges[idx][0] for idx in valid_edge_indices])
    
            # Remove already visited nodes
            target_nodes = new_target_nodes - visited_nodes
    
            # Add newly discovered nodes to visited set
            visited_nodes.update(target_nodes)
    
#        incoming_edges_with_flow = [self.both_edges[idx] for idx in incoming_edge_indices]
        action_mask = np.zeros(self.num_both_edges, dtype=bool)
        action_mask[list(incoming_edge_indices)] = True
        return(action_mask)
    
    def mask_fn(self, depth=1):
        """
        Fully vectorized function using cached flow information.
        Maximum speed optimization.
        """
        remaining_budget = self.state['budget'][0]
        edge_interdicted = self.state['edge_interdicted']
    
        action_mask = np.ones(self.action_space.n, dtype=np.float32)
    
        # All vectorized checks
        sufficient_budget = (remaining_budget - self.state['edge_costs'][:self.num_interdictable]) >= -0.1
        #has_capacity = self.state['edge_capacity'][:self.num_interdictable] > 0
        has_probability = self.state['edge_interdiction_probability'][:self.num_interdictable] > 0
    
        within_limit = (edge_interdicted[:self.num_interdictable] + 1) <= self.max_interdictions
        
        # Strategy-specific checks
        if self.attacker_strategy == 'isolate':
            # Get edges on paths from isolate objectives to sources
            on_path_to_source = self.get_edges_on_paths_to_source(start_nodes = self.state['isolate_objective'])
            #has_flow = self.cached_flow_array[:self.num_interdictable] > 0
            valid_actions = (sufficient_budget &  
                             has_probability & 
                             #has_capacity &
                             on_path_to_source &
                             #has_flow &
                             within_limit)
        
        elif self.attacker_strategy == 'canalize':
            has_flow = self.cached_flow_array[:self.num_interdictable] > 0
            not_target = self.state['canalize_objective'][:self.num_interdictable] != 1
            valid_actions = (sufficient_budget & #has_capacity & 
                             has_probability & 
                             within_limit & has_flow & not_target)
        elif self.attacker_strategy == 'divert':
            has_flow = self.cached_flow_array[:self.num_interdictable] > 0
            not_target = self.state['divert_to_objective'][:self.num_interdictable] != 1
            
            # Check if any edges on the divert_from path have been interdicted
            from_path_interdicted = np.any((self.state['edge_interdicted'][:self.num_interdictable] > 0) & 
                                         (self.state['divert_from_objective'][:self.num_interdictable] == 1))

            if not from_path_interdicted:
                is_target = self.state['divert_from_objective'][:self.num_interdictable] == 1
            else:
                is_target = True

            valid_actions = (sufficient_budget & #has_capacity & 
                             has_probability & is_target &
                             within_limit & has_flow & not_target)
        else:
            has_flow = self.cached_flow_array[:self.num_interdictable] > 0
            valid_actions = (has_flow & 
                             sufficient_budget & #self.has_capacity & 
                             self.has_probability & 
                             within_limit)
    
        action_mask[:self.num_interdictable] = valid_actions.astype(np.float32)
    
        return action_mask

@ray.remote
class _ProgressActor:
    def __init__(self):
        self.count = 0
    def increment(self, n: int = 1):
        self.count += int(n)
    def get_count(self):
        return self.count
    def reset(self):
        self.count = 0

# New: centralized memo actor (shared between driver + workers)
@ray.remote
class _SharedMemoActor:
    def __init__(self):
        self.memo = {}
    def get(self, key):
        return self.memo.get(key)
    def set(self, key, value):
        self.memo[key] = value
    def size(self):
        return len(self.memo)

@ray.remote
class _SharedAlphaActor:
    def __init__(self):
        self.alpha = -float('inf')
    def get(self):
        return self.alpha
    def update(self, new_alpha):
        if new_alpha > self.alpha:
            self.alpha = new_alpha

@ray.remote
class SharedOutcomeMemoActor:
    """
    Centralized cache for stochastic max-flow outcomes.
    Stores: (outcome_tuple, strategy) -> {'objective': val, 'flows': dict}
    """
    def __init__(self):
        self.cache = {}
    def get_batch(self, keys):
        return [self.cache.get(k) for k in keys]
    def set_batch(self, keys, values):
        for k, v in zip(keys, values):
            self.cache[k] = v
    def size(self):
        return len(self.cache)
    def clear(self):
        self.cache.clear()

@ray.remote
class _RemoteEnvWorker:
    def __init__(self, nodes, edges, seed, state_snapshot, attacker_strategy, min_edge_cost,
                 num_both_edges, deterministic_outcomes, multiple_interdiction_attempts,
                 progress_actor=None, memo_actors=None, budget_levels=1, progress_granularity=50,
                 max_depth_inner=20, outcome_memo_actor=None, outcome_memo_actors=None, alpha_actor=None):
        """
        Worker now accepts a progress_actor handle, a shared memo_actor handle,
        and budget_levels so it can estimate progress for invalid actions.
        """
        import importlib, copy, numpy as np, ray as _ray, time
        from collections import defaultdict
        env_mod = importlib.import_module("env_TA")
        CustomEnv = getattr(env_mod, "CustomEnv")

        # Instantiate env with the same key flags so internals (num_both_edges, spaces, models) are set up
        # Pass None for actors initially to avoid clearing them during load_network_from_state
        self.env = CustomEnv(nodes, edges,
                             deterministic_agent=deterministic_outcomes,
                             multiple_interdiction_attempts=multiple_interdiction_attempts,
                             attacker_strategy=attacker_strategy,
                             outcome_memo_actor=None,
                             outcome_memo_actors=None)

        # Make a deep, writable copy of the state snapshot to avoid read-only numpy arrays
        state_copy = copy.deepcopy(state_snapshot)
        if isinstance(state_copy, dict):
            for k, v in list(state_copy.items()):
                if isinstance(v, np.ndarray):
                    try:
                        v = v.copy()
                        v.setflags(write=True)
                        state_copy[k] = v
                    except Exception:
                        state_copy[k] = np.array(v, copy=True)

        # Restore state on the worker
        self.env.load_network_from_state(seed, state_copy)

        # Attach shared actors after loading state
        self.env.outcome_memo_actor = outcome_memo_actor
        self.env.outcome_memo_actors = outcome_memo_actors

        self.attacker_strategy = attacker_strategy
        self.min_edge_cost = min_edge_cost
        self.num_both_edges = num_both_edges
        self.max_depth_inner = max_depth_inner
        self.progress_actor = progress_actor
        self.alpha_actor = alpha_actor
        
        # Handle Sharded Memo Actors
        self.memo_actors = memo_actors
        self.num_memo_shards = len(memo_actors) if memo_actors else 0
        
        self.progress_granularity = int(progress_granularity)
        self.budget_levels = int(budget_levels)

    def evaluate_subtree(self, remaining_budget, interdicted_state, depth):
        import numpy as np, ray as _ray, time, zlib # Added zlib for stable hashing
        memo_local = {}
        local_counter = 0
        
        # Local cache of the global alpha value
        local_alpha_cache = -float('inf')
        if self.alpha_actor:
            try:
                local_alpha_cache = _ray.get(self.alpha_actor.get.remote())
            except Exception:
                pass

        #Time-based throttling variables
        last_report_time = time.time()
        report_interval = 0.5  # Max 2 reports per second per worker

        def maybe_flush_progress():
            nonlocal local_counter, last_report_time, local_alpha_cache
            # 1. Check if we have enough accumulated progress (batch size)
            if self.progress_actor is not None and local_counter >= self.progress_granularity:
                # 2. Check if enough time has passed (throttle)
                now = time.time()
                if (now - last_report_time) > report_interval:
                    try:
                        # report and reset local counter (best-effort)
                        self.progress_actor.increment.remote(local_counter)
                        local_counter = 0
                        last_report_time = now
                        
                        # Sync alpha
                        if self.alpha_actor:
                            remote_val = _ray.get(self.alpha_actor.get.remote())
                            if remote_val > local_alpha_cache:
                                local_alpha_cache = remote_val
                    except Exception:
                        pass
                # If time hasn't passed, we keep accumulating local_counter.
                # This efficiently batches "bursty" progress (like cache hits).

        def dp_local(rem_budget, inter_state, d, alpha=-float('inf')):
            nonlocal local_counter, local_alpha_cache
            key = inter_state[:self.num_both_edges].tobytes()
            
            # Incorporate global knowledge
            alpha = max(alpha, local_alpha_cache)

            # 1) check local cache
            if key in memo_local:
                # ADDED: Count volume for cached hit
                vol = int(int(self.num_both_edges) ** max(0, self.budget_levels - d))
                local_counter += vol
                maybe_flush_progress()
                return memo_local[key]

            # 2) check centralized memo (SHARDED)
            t_start = time.perf_counter()
            shared_val = None
            target_actor = None
            
            if self.num_memo_shards > 0:
                # Deterministic sharding based on key content using zlib.adler32 (fast & stable)
                shard_idx = zlib.adler32(key) % self.num_memo_shards
                target_actor = self.memo_actors[shard_idx]
                
                try:
                    shared_val = _ray.get(target_actor.get.remote(key))
                except Exception:
                    shared_val = None

            if shared_val is not None:
                memo_local[key] = shared_val
                # ADDED: Count volume for cached hit
                vol = int(int(self.num_both_edges) ** max(0, self.budget_levels - d))
                local_counter += vol
                maybe_flush_progress()
                return shared_val

            # save/restore small pieces of state
            old_budget = self.env.state['budget'][0]
            old_interdicted = self.env.state['edge_interdicted'].copy()

            self.env.state['budget'][0] = rem_budget
            self.env.state['edge_interdicted'][:] = inter_state

            # terminal objective
            t_start = time.perf_counter()

            # Capture flows to update environment state for mask_fn
            current_flows = None
            
            if self.attacker_strategy == "zero_sum":
                final_objective, current_flows = self.env._compute_objective_and_flows()
                final_objective = -final_objective
            elif self.attacker_strategy == 'canalize':
                final_objective, current_flows = self.env._calculate_canalize_objective_and_flows()
            elif self.attacker_strategy == 'isolate':
                final_objective, current_flows = self.env._calculate_isolate_objective_and_flows()
                final_objective = -final_objective
            elif self.attacker_strategy == 'divert':
                final_objective, current_flows = self.env._calculate_divert_objective_and_flows()
            else:
                final_objective = -float('inf')
                current_flows = {}

            # Update reference flows and cache for mask_fn
            self.env.reference_flows = current_flows
            self.env._cache_flow_array()

            # base case
            if rem_budget < self.min_edge_cost or d >= self.max_depth_inner:
                # restore
                self.env.state['budget'][0] = old_budget
                self.env.state['edge_interdicted'][:] = old_interdicted

                memo_local[key] = (final_objective, [])
                # Count the volume of the skipped subtree (leaves)
                volume = int(int(self.num_both_edges) ** max(0, self.budget_levels - d))
                local_counter += volume
                maybe_flush_progress()
                # publish to central memo (best-effort, async)
                if target_actor is not None:
                    try:
                        target_actor.set.remote(key, memo_local[key])
                    except Exception:
                        pass
                return final_objective, []

            action_mask = self.env.mask_fn(depth=d)

            # restore
            self.env.state['budget'][0] = old_budget
            self.env.state['edge_interdicted'][:] = old_interdicted
            valid_actions = np.where(action_mask[:self.num_both_edges] == 1)[0]

            # Report the discovery of invalid actions as progress using estimated subtree sizes
            num_invalid = int(self.num_both_edges) - len(valid_actions)
            if num_invalid > 0 and self.progress_actor is not None:
                try:
                    # estimated states pruned by each invalid action
                    est_per_invalid = int(int(self.num_both_edges) ** max(0, self.budget_levels - (d + 1)))
                    local_counter += num_invalid * est_per_invalid
                    maybe_flush_progress()
                except Exception:
                    pass

            if len(valid_actions) == 0:
                memo_local[key] = (final_objective, [])
                # No increment here; the invalid logic above covered the entire subtree volume
                maybe_flush_progress()
                if target_actor is not None:
                    try:
                        target_actor.set.remote(key, memo_local[key])
                    except Exception:
                        pass
                return final_objective, []

            # Initialize with current state value (allow stopping here)
            best_reward = final_objective
            best_seq = []
            
            # Update alpha with current node value
            alpha = max(alpha, best_reward)

            if self.attacker_strategy in ['zero_sum', 'isolate']:
                # Heuristic sorting for pruning
                caps = self.env.state['edge_capacity'][valid_actions]
                probs = self.env.state['edge_interdiction_probability'][valid_actions]
                costs = self.env.state['edge_costs'][valid_actions]
                
                # New Heuristic: (capacity * probability) + (self.DEFAULT_TRAINING_EDGE_CAPACITY_RANGE[1]*(remaining_budget-cost))
                max_edge_capacity = self.env.DEFAULT_TRAINING_EDGE_CAPACITY_RANGE[1]
                heuristics = (caps * probs) + (max_edge_capacity * (rem_budget - costs))
                
                # Sort descending
                sorted_indices = np.argsort(-heuristics)
                valid_actions = valid_actions[sorted_indices]
                heuristics = heuristics[sorted_indices]
            
            for i, action in enumerate(valid_actions):
                # Pruning for zero_sum
                if self.attacker_strategy in ['zero_sum', 'isolate']:
                     # Pruning condition: 
                     # current obj value - (current remaining budget * heuristic) > current best objective value
                     # -final_objective - (rem_budget * heuristics[i]) > -alpha
                     # final_objective + (rem_budget * heuristics[i]) < alpha
                     # UPDATED: Use the new heuristic directly (it already looks ahead)
                     if final_objective + heuristics[i] < alpha:
                         # Since heuristics are sorted descending, all subsequent actions will also fail
                         # Report skipped progress
                         skipped_actions = len(valid_actions) - i
                         est_per_skipped = int(self.num_both_edges ** max(0, self.budget_levels - (d + 1)))
                         local_counter += skipped_actions * est_per_skipped
                         maybe_flush_progress()
                         break

                # Apply move in-place
                inter_state[action] += 1
                new_budget = rem_budget - self.env.state['edge_costs'][action]
                
                # Recurse
                fut_reward, fut_seq = dp_local(new_budget, inter_state, d + 1, alpha)
                
                # Backtrack (Revert move)
                inter_state[action] -= 1
                
                if fut_reward > best_reward:
                    best_reward = fut_reward
                    best_seq = [action] + fut_seq
                    alpha = max(alpha, best_reward)
                    
                    # Update global alpha if we found something better
                    if alpha > local_alpha_cache:
                        local_alpha_cache = alpha
                        if self.alpha_actor:
                            self.alpha_actor.update.remote(alpha)

            memo_local[key] = (best_reward, best_seq)
            # Do NOT increment local_counter here for internal nodes
            
            # publish result to central memo (best-effort, async)
            if target_actor is not None:
                try:
                    target_actor.set.remote(key, memo_local[key])
                except Exception:
                    pass

            return best_reward, best_seq

        result = dp_local(remaining_budget, interdicted_state.copy(), depth)
        # flush any remaining progress
        if self.progress_actor is not None and local_counter > 0:
            try:
                self.progress_actor.increment.remote(local_counter)
            except Exception:
                pass
        return result

# New parallel entrypoint (keeps your old function untouched)
def solve_backward_induction_ray(self, verbose=False, n_workers=4, worker_depth=None, ray_address=None):
    """
    Parallelized backward induction using Ray with Adaptive Frontier Expansion.
    """
    # init ray if not already
    if not ray.is_initialized():
        ray.init(address=ray_address, ignore_reinit_error=True)

    # precompute min edge cost etc
    real_edge_costs = self.state['edge_costs'][:self.num_both_edges]
    self.min_edge_cost = min(real_edge_costs[real_edge_costs > 0], default=float('inf'))
    if self.min_edge_cost == float('inf'):
        self.min_edge_cost = 1

    # snapshot state to send to workers
    import copy, numpy as np, ray as _ray
    state_snapshot = copy.deepcopy(self.state)
    seed = getattr(self, 'seed', None)

    # create actors
    progress_actor = _ProgressActor.remote()
    # SHARDED MEMOIZATION
    # Create multiple memo actors to reduce lock contention
    # Using n_workers shards ensures high throughput
    num_memo_shards = min(2, n_workers) 
    memo_actors = [_SharedMemoActor.remote() for _ in range(num_memo_shards)]

    # Create outcome memoization actor ONLY if stochastic
    outcome_memo_actors = []
    if not self.deterministic_outcomes:
        num_outcome_shards = min(4, n_workers)
        outcome_memo_actors = [SharedOutcomeMemoActor.remote() for _ in range(num_outcome_shards)]
    
    # Create alpha actor
    alpha_actor = _SharedAlphaActor.remote()
    
    max_budget = self.state['budget'][0]
    budget_levels = int(max_budget // self.min_edge_cost) if self.min_edge_cost > 0 else 1

    workers = [
        _RemoteEnvWorker.remote(
            self.nodes,
            self.edges_reset,
            seed,
            state_snapshot,
            self.attacker_strategy,
            self.min_edge_cost,
            self.num_both_edges,
            self.deterministic_outcomes,
            self.multiple_interdiction_attempts,
            progress_actor=progress_actor,
            memo_actors=memo_actors, # Pass list of actors
            budget_levels=budget_levels,
            progress_granularity=2000,
            max_depth_inner=20,
            outcome_memo_actors=outcome_memo_actors,
            alpha_actor=alpha_actor
        )
        for _ in range(n_workers)
    ]

    # Setup progress bar
    budget_levels_local = budget_levels
    estimated_states = (int(self.num_both_edges) ** budget_levels_local) if budget_levels_local > 0 else 1
    
    from tqdm import tqdm
    import threading, time
    stop_event = threading.Event()
    pbar = tqdm(total=estimated_states, desc="DP States (Adaptive)", unit=" states", disable=not verbose)
    last_reported = 0

    def poll_progress():
        nonlocal last_reported
        while not stop_event.is_set():
            try:
                current = _ray.get(progress_actor.get_count.remote(), timeout=1)
            except Exception:
                current = last_reported
            delta = current - last_reported
            if delta > 0:
                pbar.update(delta)
                last_reported = current
            time.sleep(0.5)

    poll_thread = threading.Thread(target=poll_progress, daemon=True)
    poll_thread.start()

    # --- Adaptive Frontier Expansion ---
    
    # Capture num_both_edges for use inside inner class
    num_edges_limit = int(self.num_both_edges)

    class TreeNode:
        def __init__(self, budget, state, depth, parent=None, action_from_parent=None):
            self.budget = budget
            self.state = state
            self.depth = depth
            self.parent = parent
            self.action_from_parent = action_from_parent
            self.children = [] # List of TreeNodes
            self.is_terminal = False
            self.value = None
            self.best_sequence = []
            self.key = state[:num_edges_limit].tobytes()

    # Root node
    root_node = TreeNode(
        self.state['budget'][0], 
        self.state['edge_interdicted'].copy(), 
        0
    )
    
    # Frontier of nodes that need processing (either expansion or solving)
    frontier = [root_node]
    
    # Target number of tasks to generate (e.g., 4x workers ensures good balancing)
    # TWEAK THIS: Higher = more small tasks (better balancing, more overhead)
    TARGET_TASKS = n_workers * 10 
    
    # Local memoization for the driver expansion phase
    memo_driver = {}

    # 1. Expansion Phase
    # We pop nodes and expand them until we have enough tasks or run out of nodes
    tasks_to_solve = [] # Nodes ready to be sent to workers
    
    while frontier:
        # If we have enough tasks, stop expanding and move remaining frontier to solve list
        if len(frontier) + len(tasks_to_solve) >= TARGET_TASKS:
            tasks_to_solve.extend(frontier)
            frontier = []
            break
        
        node = frontier.pop(0)
        
        # --- PROGRESS REPORTING FIX ---
        # Calculate potential subtree volume for this node
        remaining_depth = max(0, budget_levels_local - node.depth)
        node_volume = int(self.num_both_edges ** remaining_depth)

        # Check memo (driver side)
        if node.key in memo_driver:
            node.value, node.best_sequence = memo_driver[node.key]
            node.is_terminal = True
            # Driver pruned this whole subtree -> Report progress
            progress_actor.increment.remote(node_volume)
            continue

        # Check base cases
        if node.budget < self.min_edge_cost or node.depth >= 20:
            node.is_terminal = True
            tasks_to_solve.append(node)
            # Sent to worker -> Worker will report progress
            continue

        # Temporarily set state to generate mask
        old_budget = self.state['budget'][0]
        old_interdicted = self.state['edge_interdicted'].copy()
        self.state['budget'][0] = node.budget
        self.state['edge_interdicted'][:] = node.state

        # Update flows and cache for mask_fn
        if self.attacker_strategy == "zero_sum":
            _, self.reference_flows = self._compute_objective_and_flows()
        elif self.attacker_strategy == 'canalize':
            _, self.reference_flows = self._calculate_canalize_objective_and_flows()
        elif self.attacker_strategy == 'isolate':
            _, self.reference_flows = self._calculate_isolate_objective_and_flows()
        elif self.attacker_strategy == 'divert':
            _, self.reference_flows = self._calculate_divert_objective_and_flows()
        
        self._cache_flow_array()
        
        action_mask = self.mask_fn(depth = node.depth)
        valid_actions = np.where(action_mask[:self.num_both_edges] == 1)[0]
        
        # Restore state
        self.state['budget'][0] = old_budget
        self.state['edge_interdicted'][:] = old_interdicted

        if len(valid_actions) == 0:
            node.is_terminal = True
            tasks_to_solve.append(node)
            # Sent to worker -> Worker will report progress
            continue
        
        # --- PROGRESS REPORTING FIX ---
        # If we expand this node, we are responsible for reporting the volume of the branches we DON'T take.
        num_invalid = int(self.num_both_edges) - len(valid_actions)
        if num_invalid > 0:
            child_remaining_depth = max(0, budget_levels_local - (node.depth + 1))
            child_volume = int(self.num_both_edges) ** child_remaining_depth
            pruned_volume = num_invalid * child_volume
            progress_actor.increment.remote(pruned_volume)

        # Expand children
        for action in valid_actions:
            new_state = node.state.copy()
            new_state[action] += 1
            new_budget = node.budget - self.state['edge_costs'][action]
            
            child = TreeNode(new_budget, new_state, node.depth + 1, parent=node, action_from_parent=action)
            node.children.append(child)
            frontier.append(child)

    # 2. Execution Phase (Dynamic Load Balancing)
    # tasks_to_solve contains the leaves of our expanded tree.
    # We send these to workers.
    
    # Prepare tasks
    # Format: (node_obj, budget, state, depth)
    pending_tasks = list(tasks_to_solve)
    
    idle_workers = list(workers)
    running_futures = {} # future -> (worker, node)
    
    while pending_tasks or running_futures:
        while idle_workers and pending_tasks:
            worker = idle_workers.pop()
            node = pending_tasks.pop(0)
            
            if node.value is not None:
                idle_workers.append(worker)
                continue
                
            future = worker.evaluate_subtree.remote(node.budget, node.state, node.depth)
            running_futures[future] = (worker, node)
        
        if running_futures:
            done_ids, _ = _ray.wait(list(running_futures.keys()), num_returns=1)
            for done_id in done_ids:
                worker, node = running_futures.pop(done_id)
                try:
                    val, seq = _ray.get(done_id)
                    node.value = val
                    node.best_sequence = seq
                    
                    # Cache result in driver memo
                    memo_driver[node.key] = (val, seq)
                except Exception as e:
                    print(f"Task failed: {e}")
                    node.value = -float('inf') # Treat as failure
                
                idle_workers.append(worker)

    # 3. Aggregation Phase (Bottom-Up)
    # We need to propagate values from leaves up to root.
    # Since we built a tree, we can do a post-order traversal or just iterate by depth reverse.
    
    # Collect all nodes in the tree
    all_nodes = []
    q = [root_node]
    while q:
        curr = q.pop(0)
        all_nodes.append(curr)
        q.extend(curr.children)
        
    # Sort by depth descending (deepest first)
    all_nodes.sort(key=lambda x: x.depth, reverse=True)
    
    for node in all_nodes:
        # Compute value of current node (stopping value)
        # Temporarily set state
        old_budget = self.state['budget'][0]
        old_interdicted = self.state['edge_interdicted'].copy()
        self.state['budget'][0] = node.budget
        self.state['edge_interdicted'][:] = node.state
        
        if self.attacker_strategy == "zero_sum":
            val, _ = self._compute_objective_and_flows()
            val = -val
        elif self.attacker_strategy == 'canalize':
            val, _ = self._calculate_canalize_objective_and_flows()
        elif self.attacker_strategy == 'isolate':
            val, _ = self._calculate_isolate_objective_and_flows()
            val = -val
        elif self.attacker_strategy == 'divert':
            val, _ = self._calculate_divert_objective_and_flows()
        else:
            val = -float('inf')
        
        self.state['budget'][0] = old_budget
        self.state['edge_interdicted'][:] = old_interdicted

        if node.children:
            # This is an internal node in our expanded tree.
            # Its value is the max of its children AND itself (stopping).
            best_val = val
            best_seq = []
            
            for child in node.children:
                # Child value should be set by now (either from worker or recursion)
                if child.value is None:
                    # Should not happen if logic is correct
                    continue
                    
                if child.value > best_val:
                    best_val = child.value
                    best_seq = [child.action_from_parent] + child.best_sequence
            
            node.value = best_val
            node.best_sequence = best_seq
            
            # Cache
            memo_driver[node.key] = (best_val, best_seq)
        
        elif node.value is None:
            # Leaf node that wasn't solved?
            node.value = val
            node.best_sequence = []
            memo_driver[node.key] = (val, [])

    # Final result
    optimal_reward = root_node.value
    optimal_sequence = root_node.best_sequence

    # Cleanup
    stop_event.set()
    poll_thread.join(timeout=2)
    try:
        final = ray.get(progress_actor.get_count.remote())
        pbar.update(final - last_reported)
    except Exception:
        pass
    pbar.close()

    for w in workers:
        try: ray.kill(w)
        except: pass
    try: ray.kill(progress_actor)
    except: pass
    
    # Kill all memo actors
    for ma in memo_actors:
        try: ray.kill(ma)
        except: pass

    for actor in outcome_memo_actors:
        try: ray.kill(actor)
        except: pass

    try: ray.kill(alpha_actor)
    except: pass

    if self.attacker_strategy in ("zero_sum", "isolate"):
        optimal_reward = -optimal_reward

    optimal_actions = [self.both_edges[idx] for idx in optimal_sequence]
    return optimal_reward, optimal_actions

# Attach the new method to CustomEnv class (if needed)
try:
    CustomEnv.solve_backward_induction_ray = solve_backward_induction_ray
except Exception:
    pass