#Purpose: Gymnasium environment for single/multiple attempt max flow deterministic/stochastic network interdiction by zero-sum/threat adaptive attackers

# Import all required packages
import os
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
from collections import defaultdict, Counter
from itertools import product
import ray
import networkx as nx
import logging # Reduce native logging noise (best-effort; affects Python loggers)
logging.getLogger("ray").setLevel(logging.WARNING)
logging.getLogger("raylet").setLevel(logging.WARNING)
# Import Solvers Mixin
from network_interdiction_solvers import InterdictionSolverMixin

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
class CustomEnv(InterdictionSolverMixin, gym.Env):
    """Custom Gym environment for network interdiction problems."""
    # Class constants
    # Set Method=1 (Dual Simplex) and MIPGap=0 for strict deterministic/optimal behavior
    # Do NOT create a global Gurobi Env at import time (not multiprocessing-safe).
    # Create a per-instance env in _initialize_maxflow_model instead.
    GUROBI_ENV = None
    
    def __init__(self, nodes, edges, deterministic_agent=True, initial_budget = None, 
                 multiple_interdiction_attempts=True, attacker_strategy="zero_sum",
                 budget_range=(0, 100), edge_capacity_range=(0, 100), 
                 edge_cost_range=(0, 10), training_budget_range=(5, 10), 
                 training_edge_capacity_range=(30, 60), training_edge_cost_range=(3, 5),
                 max_interdiction_attempts=10, max_source_flow=3, 
                 max_sink_need=3, penalty_value=-0.1, 
                 sample_size=1000, max_path_length = 6,
                 max_num_edges=500, max_num_nodes=250, 
                 objective_path_length=2,
                 outcome_memo_actor=None, outcome_memo_actors=None,
                 enable_flow_masking=True, enable_mission_masking=True):
        super(CustomEnv, self).__init__()

        #Setup core environment attributes
        self.enable_flow_masking = enable_flow_masking
        self.enable_mission_masking = enable_mission_masking
        self.last_mask_stats = {'resource': 0, 'flow': 0, 'mission': 0}
        
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
        self.objective_path_length = max(1, objective_path_length)
        self.outcome_memo_actor = outcome_memo_actor
        self.outcome_memo_actors = outcome_memo_actors
        if self.outcome_memo_actors is None and self.outcome_memo_actor is not None:
             self.outcome_memo_actors = [self.outcome_memo_actor]
        self.local_outcome_cache = {} # Add local cache here
        self.enable_outcome_caching = True # Default to True
        
        self.num_stochastic_scenarios = None
        self.num_stochastic_scenarios_IM = None

        self.max_interdictions = self.MAX_INTERDICTION_ATTEMPTS if self.multiple_interdiction_attempts else 1
        self._clear_objective_path_cache()
        
        # Initialize network structure
        self._setup_network_structure()

        # Setup observation and action spaces
        self._setup_spaces()

    def _cache_flow_array(self):
        """Fully vectorized cache using array indexing."""
        # Optimized to use pre-computed reverse keys
        flows = self.reference_flows
        
        # ADDED: Support for pre-calculated flow arrays (Optimized path)
        if isinstance(flows, np.ndarray):
            self.cached_flow_array = flows
            return

        self.cached_flow_array = np.array(
            [[flows.get(e, 0), flows.get(re, 0)] for e, re in zip(self.both_edges, self.reverse_edges_list)], 
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

        # Compute accurate distances to sink
        self._compute_node_distances()
            
    def _compute_node_distances(self):
        """Compute shortest path distance (BFS) from every node to the Super Sink."""
        # Initialize distances to infinity
        self.node_distances = {node: float('inf') for node in self.nodes}
        
        # Start BFS from Super Sink
        target = self.super_sink_nodes[0]
        self.node_distances[target] = 0
        queue = [target]
        
        while queue:
            current = queue.pop(0)
            
            # Check all incoming edges to find upstream neighbors
            # (We traverse the graph backwards from Sink -> Source)
            if current in self.edge_groups:
                for edge in self.edge_groups[current]['in']:
                    neighbor = edge[0] # The node pointing to current
                    if self.node_distances[neighbor] == float('inf'):
                        self.node_distances[neighbor] = self.node_distances[current] + 1
                        queue.append(neighbor)

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
                
        # Initialize model on first call or if it was deleted/None
        if getattr(self, 'maxflow_model', None) is None:
            self._initialize_maxflow_model()
            # Force objective re-setup
            self.strategy_objectives_setup = False

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
            self.reduce_flow_idx = self.maxflow_model.NumObj # Save the index for reduce_flow
            self.strategy_objectives_setup = True
            self.old_routing_assumption = routing_assumption
            
        # Solve and return results
        self.maxflow_model.params.Seed = 1
        
        # Reduce Flow Configuration (Robustness)
        # Check depth (number of interdictions already placed)
        current_depth = self.state['edge_interdicted'].sum()
        
        current_budget = self.state['budget'][0]
        # In case solve_max_flow is called during reset before num_interdictable is set
        limit = getattr(self, 'num_interdictable', self.num_both_edges)
        min_cost = np.min(self.state['edge_costs'][:limit]) if limit > 0 else 0
        has_budget = current_budget >= min_cost

        # Reduce Flow configuration
        # Use robustness only if we have budget for another interdiction and depth is low
        use_reduce_flow = getattr(self, 'reduce_flow', False) and has_budget

        # Keep a consistent index for the reduce flow objective so they do not stack endlessly
        reduce_flow_idx = getattr(self, 'reduce_flow_idx', self.maxflow_model.NumObj)
        
        if use_reduce_flow:
            reduce_flow_expr = grb.quicksum(self.flow_var[e] for e in self.both_edges)
            #reduce_flow_expr = grb.quicksum(self.edge_used[e] for e in self.both_edges)
            self.maxflow_model.setObjectiveN(reduce_flow_expr, index=reduce_flow_idx, priority=1, weight=-1.0, name="reduce_flow_min_edges")
        else:
            # Disable reduce_flow objective if budget not sufficient
            self.maxflow_model.setObjectiveN(0.0, index=reduce_flow_idx, priority=0, weight=0.0, name="reduce_flow_disabled")

        # Standard single solution
        self.maxflow_model.setParam('PoolSearchMode', 0)
        self.maxflow_model.setParam('PoolSolutions', 1)

        self.sensitive_edges = []
        callback = None
        if routing_assumption in ['divert', 'canalize', 'isolate']:
            self._update_sensitive_edges(routing_assumption)
            callback = self._subtour_callback

        try:
            self.maxflow_model.optimize(callback)
        except Exception as e:
            raise
        
        try:
            status = self.maxflow_model.Status
        except Exception:
            status = None
        if status == grb.GRB.OPTIMAL:
            # Use strict values without rounding to avoid flipping behavior near .5 boundaries
            # In multi-objective, trust the flow variable on super edge for the max flow value
            if self.maxflow_model.NumObj > 1:
                obj_val = self.flow_var[self.super_edge].X
            else:
                 obj_val = self.maxflow_model.ObjVal
            
            flow_results = {e: var.X for e, var in self.flow_var.items()}
            self.flow_histories = [flow_results.copy()]

            # --- NEW INJECTED CORE FLOW EXTRACTOR LOGIC ---
            if getattr(self, 'core_flow_extractor', False) and current_budget > 5:
                action_mask = self.mask_fn()
                valid_actions_indices = np.where(action_mask[:self.num_interdictable] == 1)[0]
                valid_action_edges = {self.both_edges[idx] for idx in valid_actions_indices if idx < len(self.both_edges)}

                active_edges = {e for e in valid_action_edges if flow_results.get(e, 0) + flow_results.get((e[1], e[0]), 0) > 1e-4}
                prev_active_edges = None
                
                iteration = 0
                while iteration < 20 and active_edges and active_edges != prev_active_edges:
                    prev_active_edges = active_edges.copy()
                    iteration += 1

                    valid_active_edges = [e for e in active_edges if e in self.flow_var and (e[1], e[0]) in self.flow_var]
                    if not valid_active_edges:
                        break

                    # Overwrite the reduce_flow objective (Index 3, Priority 1) to iteratively squeeze active elements
                    squeeze_expr = grb.quicksum(self.flow_var[e] + self.flow_var[(e[1], e[0])] for e in valid_active_edges)
                    self.maxflow_model.setObjectiveN(-squeeze_expr, index=reduce_flow_idx, priority=1, weight=1.0, name="core_flow_extractor_min_flow")
                    
                    self.maxflow_model.optimize(callback)
                    
                    if self.maxflow_model.Status not in [grb.GRB.OPTIMAL, grb.GRB.SUBOPTIMAL]:
                        break

                    # Get new flow dict
                    current_iter_flows = {e: var.X for e, var in self.flow_var.items()}
                    self.flow_histories.append(current_iter_flows)

                    # Next active edges
                    active_edges = {e for e in valid_active_edges if current_iter_flows.get(e, 0) + current_iter_flows.get((e[1], e[0]), 0) > 1e-4}

                # Restore original reduce_flow objective state for future calls
                if use_reduce_flow:
                    self.maxflow_model.setObjectiveN(reduce_flow_expr, index=reduce_flow_idx, priority=1, weight=-1.0, name="reduce_flow_min_edges")
                else:
                    self.maxflow_model.setObjectiveN(0.0, index=reduce_flow_idx, priority=0, weight=0.0, name="reduce_flow_disabled")
            
            # --- CALCULATE 1D CORE FLOW ARRAY RIGHT HERE ---
            if getattr(self, 'core_flow_extractor', False):
                core_array = np.zeros(self.num_both_edges, dtype=np.float32)
                for idx, e in enumerate(self.both_edges):
                    min_f = float('inf')
                    for hist in self.flow_histories:
                        f = hist.get(e, 0) + hist.get((e[1], e[0]), 0)
                        if f < min_f:
                            min_f = f
                    core_array[idx] = min_f
                self._current_core_flow_array = core_array
            # ---------------------------------
            
        else:
            obj_val = 0
            flow_results = {e: 0 for e in self.flow_var.keys()}
            self.flow_histories = [flow_results.copy()]
        return obj_val, flow_results 

    def _update_sensitive_edges(self, routing_assumption):
        """Identify edges that are part of the current strategy's objective."""
        self.sensitive_edges = []
        if routing_assumption == 'isolate':
            indices = np.where(self.state['isolate_objective'][:self.num_both_edges] == 1)[0]
            self.sensitive_edges = [self.both_edges[i] for i in indices]
        elif routing_assumption == 'canalize':
            self.sensitive_edges = self._get_objective_path_edges('canalize_objective')
        elif routing_assumption == 'divert':
            self.sensitive_edges =self._get_objective_path_edges('divert_from_objective') + self._get_objective_path_edges('divert_to_objective')

    def _subtour_callback(self, model, where):
        """Callback to eliminate subtours efficiently."""
        if where == grb.GRB.Callback.MIPSOL:
                
            vals = model.cbGetSolution(self.edge_used)
            # Filter edges with value > 0.5
            selected_edges = [e for e, v in vals.items() if v > 0.5]
            
            # Detect cycles involving sensitive edges
            cycles = self._find_cycles(selected_edges)
            
            for cycle in cycles:
                # Add lazy constraint: sum(edge_used in cycle) <= len(cycle) - 1
                expr = grb.quicksum(self.edge_used[e] for e in cycle)
                model.cbLazy(expr <= len(cycle) - 1)

    def _find_cycles(self, edges):
        """Find cycles in solution graph using targeted BFS on sensitive edges."""
        if not hasattr(self, 'sensitive_edges') or not self.sensitive_edges:
             return []
                     
        active_edges_set = set(edges)
        active_sensitive = [e for e in self.sensitive_edges if e in active_edges_set]
        cycles = []
        
        G = nx.DiGraph()
        G.add_edges_from(edges)
            
        for u, v in active_sensitive:
            if len(cycles) >= 20: break
            try:
                # Find shortest path from v back to u to complete the cycle
                path = nx.shortest_path(G, source=v, target=u)
                if len(path) <= 20: # Limiting length to match old behavior
                    cycle_edges = [(u, v)]
                    for i in range(len(path) - 1):
                        cycle_edges.append((path[i], path[i+1]))
                    cycles.append(cycle_edges)
            except nx.NetworkXNoPath:
                continue
        return cycles 

    def _initialize_maxflow_model(self):
        """Initialize the Gurobi max flow model with variables and constraints."""
        # Ensure a per-instance Gurobi environment (safe for multiprocessing workers)
        if getattr(self, 'GUROBI_ENV', None) is None:
            try:
                # Added TimeLimit and MIPGap to prevent the solver from hanging on proving optimality
                self.GUROBI_ENV = grb.Env(params={"OutputFlag": 0, "LogToConsole": 0, "Threads": 2, "Seed": 1, "TimeLimit": 60, "MIPGap": 0.00001, "MIPFocus": 1})
            except Exception as e:
                self.GUROBI_ENV = None
        try:
            self.maxflow_model = grb.Model("Max Flow", env=self.GUROBI_ENV)
        except Exception as e:
            raise
        self.super_edge = (self.max_num_nodes, 1)
        
        # Prepare edge list with super sink-source connection
        self.mf_all_both_edges = self.all_both_edges + [self.super_edge] 

        ##VARIABLES
        # Add Flow variables
        self.flow_var = self.maxflow_model.addVars(self.mf_all_both_edges, vtype=grb.GRB.CONTINUOUS, lb=0, name="flow_var")

        # Add Edge Usage variables
        self.edge_used = self.maxflow_model.addVars(self.all_both_edges, vtype=grb.GRB.BINARY, name="edge_used")

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
        #self.maxflow_model.addConstrs((self.flow_var[e] >= self.edge_used[e] for e in self.all_both_edges), name="min_flow_forward")

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
             if isinstance(self.forward_cons, dict) and not isinstance(self.forward_cons, grb.tupledict):
                 for c in self.forward_cons.values(): self.maxflow_model.remove(c)
                 for c in self.reverse_cons.values(): self.maxflow_model.remove(c)
             else:
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
        Update capacity constraints using Variable Upper Bounds (UB).
        Optimized logic:
        1. Static constraints enforce directionality: Flow[e] <= MaxCap * Used[e]
        2. Dynamic UB enforces capacity: Flow[e] <= CurrentCap
        
        This avoids creating/modifying linear constraints in Python loops entirely during updates.
        """
        # 1. Initialize Static Constraints (One-time setup)
        if not getattr(self, 'static_capacity_constraints_setup', False):
            # Clean up old constraints if they exist
            if hasattr(self, 'forward_cons'):
                 if isinstance(self.forward_cons, dict) and not isinstance(self.forward_cons, grb.tupledict):
                     for c in self.forward_cons.values(): self.maxflow_model.remove(c)
                     for c in self.reverse_cons.values(): self.maxflow_model.remove(c)
                 else:
                     try: self.maxflow_model.remove(self.forward_cons)
                     except: pass
                     try: self.maxflow_model.remove(self.reverse_cons)
                     except: pass
            
            self.forward_cons = {}
            self.reverse_cons = {}

            # Create Static Directionality Constraints (Big-M style)
            # We use the max possible capacity to ensure this never restricts valid flow
            # The strict capacity will be enforced by the Variable UB.
            max_cap = float(self.DEFAULT_EDGE_CAPACITY_RANGE[1])
            if max_cap <= 0: max_cap = 100.0 # Safety fallback

            # Create constraints for all edges once
            for e in self.both_edges:
                self.maxflow_model.addConstr(
                    self.flow_var[e] - max_cap * self.edge_used[e] <= 0,
                    name=f"static_dir_fwd"
                )
                self.maxflow_model.addConstr(
                    self.flow_var[(e[1], e[0])] - max_cap * self.edge_used[(e[1], e[0])] <= 0,
                    name=f"static_dir_rev"
                )
            
            self.static_capacity_constraints_setup = True
            self.maxflow_model.update()

        # 2. Batch Update Variable Upper Bounds
        # This is extremely fast (O(1) python call wrapping C loop)
        
        # Prepare lists aligned with self.both_edges
        edges_list = self.both_edges
        
        # Collect variables
        fwd_vars = [self.flow_var[e] for e in edges_list]
        rev_vars = [self.flow_var[(e[1], e[0])] for e in edges_list]
        
        # Collect new capacities (aligned with edges_list)
        new_caps = [capacity_dict.get(e, 0) for e in edges_list]
        
        # Apply Batch Update
        self.maxflow_model.setAttr("UB", fwd_vars, new_caps)
        self.maxflow_model.setAttr("UB", rev_vars, new_caps)      
        
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

        if routing_assumption in ['divert', 'canalize', 'isolate']:
            # Use subtour elimination callback - requires BINARY vars
            self.maxflow_model.params.LazyConstraints = 1
        else:
            self.maxflow_model.params.LazyConstraints = 0

        if routing_assumption == "zero_sum":
            pass

        elif routing_assumption == "isolate":
            # Secondary: Maximize flow to isolated edges (Priority 5)
            target_indices = np.where(self.state['isolate_objective'][:self.num_both_edges] == 1)[0]
            target_edges = [self.both_edges[i] for i in target_indices]
            
            if target_edges:
                # Only count flow towards the sink (encoded in the edge tuple)
                expr = grb.quicksum(self.flow_var[e] for e in target_edges)
                self.maxflow_model.setObjectiveN(expr, index=1, priority=5, weight=1.0, name="max_isolate_flow")

        elif routing_assumption == "canalize":
            # Secondary: Minimize the minimum forward flow through the canalized path (Priority 5)
            ordered_fwd_edges = self._get_objective_path_edges('canalize_objective')
            
            if ordered_fwd_edges:
                # 1. Forward Direction Min Flow
                z_fwd = self.maxflow_model.addVar(vtype=grb.GRB.CONTINUOUS, name="min_canalize_fwd")
                self.aux_vars.append(z_fwd)
                
                path_flow_vars_fwd = [self.flow_var[e] for e in ordered_fwd_edges]
                gc_fwd = self.maxflow_model.addGenConstrMin(z_fwd, path_flow_vars_fwd, name="min_flow_gc_fwd")
                self.aux_constrs.append(gc_fwd)
                
                # Minimize z_fwd (since ModelSense is MAXIMIZE, weight=-1.0)
                self.maxflow_model.setObjectiveN(z_fwd, index=1, priority=5, weight=-1.0, name="min_canalize_fwd_obj")

        elif routing_assumption == "divert":
            # Identify Edge Sets
            from_edges = self._get_objective_path_edges('divert_from_objective')
            to_edges = self._get_objective_path_edges('divert_to_objective')

            # Determine Phase based on existence of reference flows
            if hasattr(self, 'reference_start_flows') and self.reference_start_flows is not None:
                # INTERDICTION PHASE: Minimize min(A - Min(F), Min(T) - B)
                A = self.reference_start_flows[0] # Reference min flow on 'from' path
                B = self.reference_start_flows[1] # Reference min flow on 'to' path
                
                if from_edges and to_edges:
                    # Define z_from = Min(set F)
                    z_from = self.maxflow_model.addVar(vtype=grb.GRB.CONTINUOUS, name="z_from_F")
                    self.aux_vars.append(z_from)
                    path_flow_vars_from = [self.flow_var[e] for e in from_edges]
                    gc_from = self.maxflow_model.addGenConstrMin(z_from, path_flow_vars_from, name="gc_min_F")
                    self.aux_constrs.append(gc_from)

                    # Define z_to = Min(set T)
                    z_to = self.maxflow_model.addVar(vtype=grb.GRB.CONTINUOUS, name="z_to_T")
                    self.aux_vars.append(z_to)
                    path_flow_vars_to = [self.flow_var[e] for e in to_edges]
                    gc_to = self.maxflow_model.addGenConstrMin(z_to, path_flow_vars_to, name="gc_min_T")
                    self.aux_constrs.append(gc_to)

                    # Terms: term1 = A - z_from, term2 = z_to - B
                    term1 = self.maxflow_model.addVar(lb=-grb.GRB.INFINITY, vtype=grb.GRB.CONTINUOUS, name="term1")
                    term2 = self.maxflow_model.addVar(lb=-grb.GRB.INFINITY, vtype=grb.GRB.CONTINUOUS, name="term2")
                    self.aux_vars.extend([term1, term2])

                    c1 = self.maxflow_model.addConstr(term1 == A - z_from, name="c_term1")
                    c2 = self.maxflow_model.addConstr(term2 == z_to - B, name="c_term2")
                    self.aux_constrs.extend([c1, c2])

                    # Objective: Minimize min(term1, term2) -> Weight -1.0
                    obj_divert = self.maxflow_model.addVar(lb=-grb.GRB.INFINITY, vtype=grb.GRB.CONTINUOUS, name="obj_divert")
                    self.aux_vars.append(obj_divert)

                    gc_obj = self.maxflow_model.addGenConstrMin(obj_divert, [term1, term2], name="gc_obj_divert")
                    self.aux_constrs.append(gc_obj)

                    self.maxflow_model.setObjectiveN(obj_divert, index=1, priority=5, weight=-1.0, name="min_divert_metric")

            else:
                # INITIALIZATION PHASE: Maximize (Min(F) - Min(T))
                # Establish strong flow on F and empty T
                if from_edges and to_edges:
                    # Define z_from = Min(set F)
                    z_from = self.maxflow_model.addVar(vtype=grb.GRB.CONTINUOUS, name="z_from_init")
                    self.aux_vars.append(z_from)
                    path_flow_vars_from = [self.flow_var[e] for e in from_edges]
                    gc_from = self.maxflow_model.addGenConstrMin(z_from, path_flow_vars_from, name="gc_min_F_init")
                    self.aux_constrs.append(gc_from)

                    # Define z_to = Min(set T)
                    z_to = self.maxflow_model.addVar(vtype=grb.GRB.CONTINUOUS, name="z_to_init")
                    self.aux_vars.append(z_to)
                    path_flow_vars_to = [self.flow_var[e] for e in to_edges]
                    gc_to = self.maxflow_model.addGenConstrMin(z_to, path_flow_vars_to, name="gc_min_T_init")
                    self.aux_constrs.append(gc_to)
                    
                    # Objective: z_from - z_to
                    diff = self.maxflow_model.addVar(lb=-grb.GRB.INFINITY, vtype=grb.GRB.CONTINUOUS, name="init_diff")
                    self.aux_vars.append(diff)
                    c = self.maxflow_model.addConstr(diff == z_from - z_to)
                    self.aux_constrs.append(c)
                    
                    # Maximize diff (Weight 1.0)
                    self.maxflow_model.setObjectiveN(diff, index=1, priority=5, weight=1.0, name="max_init_diff")

                elif from_edges:
                    # Fallback
                    z_from = self.maxflow_model.addVar(vtype=grb.GRB.CONTINUOUS, name="min_from_flow")
                    self.aux_vars.append(z_from)
                    
                    constrs = self.maxflow_model.addConstrs(
                        (z_from <= self.flow_var[e] for e in from_edges),
                        name="min_from_constr"
                    )
                    self.aux_constrs.extend(constrs.values())
                    self.maxflow_model.setObjectiveN(z_from, index=1, priority=5, weight=1.0, name="max_min_divert_from")

        self.maxflow_model.update()
    
    # BEGIN Gymnasium Environment Methods        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state and return observation."""
        # Clean up any existing models
        self._cleanup_models()
        self.strategy_objectives_setup = False
        self.old_routing_assumption = None
        self.reference_start_flows = None
        self.reference_start_flow = None
        self.reference_obj = 0.0
        self.reference_budget = 0.0

        # Clear local outcome cache on reset because capacities/objectives change
        self.local_outcome_cache = {}
        self._clear_objective_path_cache(('canalize_objective', 'divert_from_objective', 'divert_to_objective'))

        # Kill centralized outcome cache actors if they exist to free memory completely
        if self.outcome_memo_actors:
            for actor in self.outcome_memo_actors:
                ray.kill(actor)
            self.outcome_memo_actors = None
        elif self.outcome_memo_actor:
            ray.kill(self.outcome_memo_actor)
            self.outcome_memo_actor = None
        
        # Call parent reset and set random seeds
        super().reset(seed=seed)
        if seed is not None:
            self._set_random_seeds(seed)

        while True:
            # Generate network parameters
            # Sample edge capacities
            raw_capacities = self.base_spaces['edge_capacity'].sample()[:self.num_both_edges]
            edge_capacities = ((raw_capacities / 100.0) * (self.DEFAULT_TRAINING_EDGE_CAPACITY_RANGE[1]-self.DEFAULT_TRAINING_EDGE_CAPACITY_RANGE[0]) + self.DEFAULT_TRAINING_EDGE_CAPACITY_RANGE[0]).astype(int)
            if self.MAX_SOURCE_FLOW is not None:
                edge_capacities[self.super_source_out_indices] = self.DEFAULT_TRAINING_EDGE_CAPACITY_RANGE[0] * np.random.uniform(0.85, self.MAX_SOURCE_FLOW)
            if self.MAX_SINK_NEED is not None:
                edge_capacities[self.super_sink_in_indices] = self.DEFAULT_TRAINING_EDGE_CAPACITY_RANGE[0] * np.random.uniform(0.85, self.MAX_SINK_NEED)
            
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
            if self.attacker_strategy == 'canalize':
                self._cache_objective_path('canalize_objective')
            elif self.attacker_strategy == 'divert':
                self._cache_objective_paths(('divert_from_objective', 'divert_to_objective'))

            # Calculate reference objective value for the attacker's strategy
            is_valid_network = self._initialize_strategy_references(is_reset_loop=True)
            if not is_valid_network:
                continue
            
            # If we made it here, the environment is valid
            break
        
        self.reference_budget = remaining_budget[0]
        
        return self.state, {}
    
    def _cleanup_models(self):
        """Clean up any existing Gurobi models to free resources."""
        models_to_cleanup = ['master_model', 'sub_model', 'optimal_stochastic_model', 'optimal_stochastic_model_IM', 'maxflow_model']
        
        # Reset optimizations and model flags
        self.static_capacity_constraints_setup = False
        self.strategy_objectives_setup = False
        self.old_routing_assumption = None
        self.num_stochastic_scenarios = None
        self.num_stochastic_scenarios_IM = None
    
        for model_name in models_to_cleanup:
            if hasattr(self, model_name):
                model = getattr(self, model_name)
                # Dispose model if it exists
                if model is not None:
                    try:
                        model.dispose()
                    except Exception:
                        pass
                # Remove attribute
                try:
                    delattr(self, model_name)
                except Exception:
                    try:
                        setattr(self, model_name, None)
                    except Exception:
                        pass
    
        # Consolidate related attributes to clean up
        cleanup_attrs = [
            # Stochastic Benders attributes
            'benders_cuts', 
            'stochastic_gamma', 'stochastic_gamma_constr', 'stochastic_budget_constr', 'stochastic_alpha', 'stochastic_beta', 'stochastic_source_sink_constr', 'stochastic_aabg_constr', 'stochastic_aabg_reverse_constr', 'stochastic_old_state', 'stochastic_old_interdicted_edges',
            # Stochastic IM attributes
            'stochastic_gamma_IM', 'stochastic_gamma_constr_IM', 'stochastic_budget_constr_IM', 'stochastic_alpha_IM', 'stochastic_beta_IM', 'stochastic_source_sink_constr_IM', 'stochastic_aabg_constr_IM', 'stochastic_aabg_reverse_constr_IM', 'stochastic_old_state_IM', 'stochastic_old_interdicted_edges_IM', 'stochastic_old_interdicted_quantities_IM',
            # Maxflow/Solution attributes
            'flow_var', 'edge_used', 'forward_cons', 'reverse_cons', 'mf_all_both_edges',
            'aux_vars', 'aux_constrs', 'sensitive_edges', 'reference_flows', 'cached_flow_array'
        ]
    
        for attr in cleanup_attrs:
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    pass

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

    def _initialize_strategy_references(self, is_reset_loop=False):
        """Initializes reference objectives, flows, and potentials based on the loaded state and strategy."""
        if self.attacker_strategy == 'zero_sum':
            self.reference_obj, self.reference_flows = self._compute_objective_and_flows()
            self.current_potential = 0.0
            
        elif self.attacker_strategy == 'canalize':
            _, flows = self.solve_max_flow(routing_assumption='canalize')
            self.reference_flows_dict = flows
            self.reference_start_flow = self._calculate_target_path_flow(flows, 'canalize_objective')
            
            self.max_canalize_objective = self._calculate_max_objective_potential('canalize_objective', self.reference_start_flow)

            if self.max_canalize_objective < 10:
                if is_reset_loop: return False
                print("Warning: Max canalize objective is very low, may lead to uninformative episode. Consider regenerating state.")
            
            self.current_potential = self._calculate_canalize_potential(flows)
            self.reference_obj = self.reference_start_flow
            self.reference_flows = flows
            
        elif self.attacker_strategy == 'isolate':
            self.reference_obj, self.reference_flows = self._calculate_isolate_objective_and_flows()
            self.current_potential = 0.0
            
        elif self.attacker_strategy == 'divert':
            _, self.reference_flows_dict = self.solve_max_flow(routing_assumption='divert')
            from_flow = self._calculate_target_path_flow(self.reference_flows_dict, 'divert_from_objective')
            to_flow = self._calculate_target_path_flow(self.reference_flows_dict, 'divert_to_objective')
            self.reference_start_flows = (from_flow, to_flow)

            self.strategy_objectives_setup = False

            self.max_divert_to_objective = self._calculate_max_objective_potential('divert_to_objective', to_flow)
            self.max_divert_objective = min(self.max_divert_to_objective, from_flow) 

            self.reference_obj = 0
            self.current_potential = self._calculate_divert_potential(self.reference_flows_dict)
            self.reference_flows = self._flows_dict_to_array(self.reference_flows_dict)
                
            if self.reference_start_flows[0] == 0 or np.sum(self.state['divert_to_objective']) < 2:
                if is_reset_loop: return False
                print("Warning: Divert strategy has no initial flow to divert or insufficient target edges, may lead to uninformative episode. Consider regenerating state.")
        
        # Standardize trackers and caches
        self.last_obj = self.reference_obj
        self._cache_flow_array()

        self.num_interdictable = min(self.num_both_edges, getattr(self.action_space, 'n', self.num_both_edges))
        self.has_probability = self.state['edge_interdiction_probability'][:self.num_interdictable] > 0
        self.has_capacity = self.state['edge_capacity'][:self.num_interdictable] > 0
        
        return True

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
        # Temporarily set state so solve_max_flow can read capacities
        self.state = base_state

        # 1. Determine max flow path
        _, flows = self.solve_max_flow()
        max_flow_edge_set = self._extract_max_flow_path(flows)
        
        # 2. Identify nodes to avoid (all nodes on max flow path)
        avoid_nodes = set()
        for u, v in max_flow_edge_set:
            avoid_nodes.add(u)
            avoid_nodes.add(v)
            
        # 3. Find candidate start nodes
        # Must not be in avoid_nodes and must have enough distance to sink 
        valid_start_nodes = [
            n for n in self.nodes 
            if n not in avoid_nodes and self.node_distances.get(n, float('inf')) >= self.objective_path_length
        ]
        
        random.shuffle(valid_start_nodes)
        
        best_path = None
        
        # 4. Attempt to find a 2-edge segment approaching the sink
        for start_node in valid_start_nodes:
            breakpoint_dist = self.node_distances.get(start_node)
            
            # Repurpose the divert path generator to compute the canalize segment
            candidate_path = self._find_alternate_segment(
                start_node=start_node, 
                divert_from_edges=[], # No specific edges to avoid other than nodes
                avoid_nodes=avoid_nodes, 
                breakpoint_dist=breakpoint_dist
            )
            
            if candidate_path:
                best_path = candidate_path
                break
                
        # 5. Fallback: if no valid disjoint path found (e.g., severe bottlenecks), use max_flow_path
        if not best_path and max_flow_edge_set:
            # Re-validate max flow edges against environment indices
            valid_mf_edges = [edge for edge in max_flow_edge_set if edge in self.edge_to_index or (edge[1], edge[0]) in self.edge_to_index]
            
            if valid_mf_edges:
                if len(valid_mf_edges) >= self.objective_path_length:
                    start_idx = random.randint(0, len(valid_mf_edges) - self.objective_path_length)
                    best_path = valid_mf_edges[start_idx : start_idx + self.objective_path_length]
                else:
                    best_path = valid_mf_edges

        # 6. Set Final Objective
        final_objective = np.zeros(self.max_num_edges, dtype=int)
        
        if best_path:
            for edge in best_path:
                # Coerce list/ndarray edges to tuple to allow dict membership checks
                if not isinstance(edge, tuple):
                    try:
                        edge = tuple(edge)
                    except Exception:
                        pass
                
                # Mark ONLY forward (directed) edge
                if edge in self.edge_to_index:
                    final_objective[self.edge_to_index[edge]] = 1
                elif (edge[1], edge[0]) in self.edge_to_index:
                    final_objective[self.edge_to_index[(edge[1], edge[0])]] = 1

        return {**base_state, 'canalize_objective': final_objective}

    def _find_random_path_from_supersource(self, avoid_nodes, max_length):
        """Find a random path from SuperSource to SuperSink avoiding specific nodes."""
        path_edges = []
        current_node = self.super_source_nodes[0]
        visited = {current_node} | avoid_nodes
        sink = self.super_sink_nodes[0]
        max_steps = 100
        steps = 0
        while current_node != sink:
            steps += 1
            if steps > max_steps:
                return False
            valid_edges = []
            if current_node in self.edge_groups:
                for edge in self.edge_groups[current_node]['out']:
                    neighbor = edge[1]
                    dist_neighbor = self.node_distances.get(neighbor, float('inf'))
                    
                    if neighbor not in visited and dist_neighbor < float('inf'):
                         valid_edges.append(edge)
            if not valid_edges:
                return False
            # Choose next
            selected_edge = random.choice(valid_edges)
            path_edges.append(selected_edge)
            visited.add(selected_edge[1])
            current_node = selected_edge[1]
            if len(path_edges) > max_length:
                return False
        return path_edges

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

    def _extract_directed_path_edges(self, objective_key):
        """Helper logic to extract path edges in order based on an objective key."""
        cache_entry = (getattr(self, 'objective_path_cache', {}) or {}).get(objective_key)
        if cache_entry is not None:
            return list(cache_entry.get('edges', ()) or ())

        obj_mask = self.state[objective_key]
        indices = np.where(obj_mask[:self.num_both_edges] == 1)[0]
        if len(indices) == 0:
            return []
        # Build set of target edges (as tuples)
        target_edges_set = {self.both_edges[i] for i in indices}

        # Count occurrences of each node across all edges
        node_counts = Counter()
        nodes = set()
        for u, v in target_edges_set:
            node_counts[u] += 1
            node_counts[v] += 1
            nodes.add(u); nodes.add(v)

        # Find endpoints: nodes that appear exactly once
        endpoints = [n for n, c in node_counts.items() if c == 1]
        
        # Helper to get distance safely
        def get_dist(n):
            return self.node_distances.get(n, -1) if hasattr(self, 'node_distances') else -1

        if len(endpoints) >= 2:
            # Choose node furthest from sink (highest distance) as start
            # This ensures we follow flow direction towards sink
            start_node = max(endpoints, key=get_dist)
            
            # Choose node closest to sink (lowest distance) as end
            # Filter remaining endpoints to find best end
            remaining = [n for n in endpoints if n != start_node]
            if remaining:
                 end_node = min(remaining, key=get_dist)
            else:
                 # Should not happen if len >= 2 and start_node is in endpoints
                 end_node = endpoints[0]
        else:
            # Fallback: pick furthest node as start and closest as end
            if not nodes:
                return []
            start_node = max(nodes, key=get_dist)
            end_node = min(nodes, key=get_dist)

        # Build undirected adjacency for traversal (so orientation stored in target_edges_set can be reversed)
        adj = defaultdict(list)
        for u, v in target_edges_set:
            adj[u].append(v)
            adj[v].append(u)

        # Traverse from start_node to end_node, avoiding immediate backtracking
        path_edges = []
        curr = start_node
        prev = None
        max_steps = len(nodes) + 5
        steps = 0
        while curr != end_node and steps < max_steps:
            steps += 1
            neighbors = adj.get(curr, [])
            # Choose next neighbor that is not the previous node if possible
            next_node = None
            for nb in neighbors:
                if nb == prev:
                    continue
                next_node = nb
                break
            # If we couldn't avoid backtrack, allow it (graph may be small)
            if next_node is None and neighbors:
                next_node = neighbors[0]

            if next_node is None:
                break

            path_edges.append((curr, next_node))
            
            prev, curr = curr, next_node

        return path_edges

    def _clear_objective_path_cache(self, objective_keys=None):
        """Clear cached directed objective path data for one or more objectives."""
        if objective_keys is None:
            objective_keys = ('canalize_objective', 'divert_from_objective', 'divert_to_objective')

        if not hasattr(self, 'objective_path_cache') or self.objective_path_cache is None:
            self.objective_path_cache = {}

        for objective_key in objective_keys:
            self.objective_path_cache[objective_key] = None

    def _cache_objective_path(self, objective_key):
        """Build and store the directed path for a single objective."""
        if not hasattr(self, 'state') or self.state is None or objective_key not in self.state:
            self._clear_objective_path_cache((objective_key,))
            return

        # Force recomputation if stale data already exists.
        self.objective_path_cache[objective_key] = None

        path_edges = self._extract_directed_path_edges(objective_key)
        ordered_edges = []
        ordered_indices = []
        ordered_nodes = []

        for edge in path_edges:
            if not isinstance(edge, tuple):
                try:
                    edge = tuple(edge)
                except Exception:
                    continue

            if edge in self.edge_to_index:
                canonical_edge = edge
            elif (edge[1], edge[0]) in self.edge_to_index:
                canonical_edge = (edge[1], edge[0])
            else:
                continue

            # FIX: Preserve the actual flow direction for solver and array evaluation
            ordered_edges.append(edge)
            
            # Use canonical strictly for objective state masking indices
            ordered_indices.append(self.edge_to_index[canonical_edge])
            if not ordered_nodes:
                ordered_nodes.append(edge[0])
            ordered_nodes.append(edge[1])

        self.objective_path_cache[objective_key] = {
            'edges': tuple(ordered_edges),
            'indices': np.asarray(ordered_indices, dtype=int),
            'lookup': {edge: idx for edge, idx in zip(ordered_edges, ordered_indices)},
            'edge_set': set(ordered_edges),
            'nodes': tuple(ordered_nodes),
        }

    def _cache_objective_paths(self, objective_keys):
        """Build and store directed paths for a collection of objectives."""
        for objective_key in objective_keys:
            self._cache_objective_path(objective_key)

    def _get_objective_path_edges(self, objective_key):
        """Return cached directed path edges for one objective, rebuilding if needed."""
        cache_entry = (getattr(self, 'objective_path_cache', {}) or {}).get(objective_key)
        if cache_entry is None:
            self._cache_objective_path(objective_key)
            cache_entry = (getattr(self, 'objective_path_cache', {}) or {}).get(objective_key)

        if cache_entry is None:
            return []
        return list(cache_entry.get('edges', ()) or ())

    def _add_divert_components(self, base_state):
        """Add divert-specific objectives to state."""
        # Temporarily set state for max flow calculation
        temp_state = {**base_state, 'divert_from_objective': np.zeros(self.max_num_edges),
                      'divert_to_objective': np.zeros(self.max_num_edges)}
        self.state = temp_state

        # Find max flow path
        _, flows = self.solve_max_flow()

        # Get edges with positive flow
        edge_flows = [(edge, flows.get(edge, 0)) for edge in self.both_edges if flows.get(edge, 0) > 0]
        # Sort by flow descending
        edge_flows.sort(key=lambda x: x[1], reverse=True)
        
        valid_start_edges = []
        # Calculate safe distance buffer for dynamic length
        min_dist = max(3, self.objective_path_length + 1) 
        
        for edge_tuple, flow_val in edge_flows:
            # Both endpoints of the edge chosen must have sufficient distance from the sink
            dist0 = self.node_distances.get(edge_tuple[0], float('inf'))
            dist1 = self.node_distances.get(edge_tuple[1], float('inf'))
            if dist0 >= min_dist and dist1 >= min_dist:
                valid_start_edges.append(edge_tuple)
        
        top_edges = valid_start_edges[:5]
        
        candidates = []
        for start_edge in top_edges:
            dist0 = self.node_distances.get(start_edge[0], float('inf'))
            dist1 = self.node_distances.get(start_edge[1], float('inf'))
            
            # The breakpoint should be the endpoint with the shorter distance
            # If tied, use the endpoint with the higher value (node ID)
            if dist0 < dist1:
                breakpoint_node = start_edge[0]
                breakpoint_dist = dist0
            elif dist1 < dist0:
                breakpoint_node = start_edge[1]
                breakpoint_dist = dist1
            else:
                breakpoint_node = max(start_edge[0], start_edge[1])
                breakpoint_dist = dist0
            
            # Trace two more edges downstream following max flow
            divert_from_segments = [start_edge]
            curr = breakpoint_node
            valid_trace = True
            
            visited_from_edges = {start_edge, (start_edge[1], start_edge[0])}
            visited_from_nodes = {start_edge[0], start_edge[1]}
            
            for _ in range(self.objective_path_length):
                out_edges = self.edge_groups.get(curr, {}).get('out', [])
                valid_out = [e for e in out_edges if e in self.both_edges and e not in visited_from_edges and e[1] not in visited_from_nodes]
                
                if not valid_out:
                    valid_trace = False
                    break
                next_edge = max(valid_out, key=lambda e: flows.get(e, 0) + random.random()*1e-6)
                divert_from_segments.append(next_edge)
                visited_from_edges.add(next_edge)
                visited_from_edges.add((next_edge[1], next_edge[0]))
                visited_from_nodes.add(next_edge[1])
                curr = next_edge[1]
            
            # Ensure we captured the start edge + the required length
            if not valid_trace or len(divert_from_segments) < self.objective_path_length + 1:
                continue

            # Nodes to avoid for divert_to (other than nodes on first edge)
            avoid_nodes = set()
            for edge in divert_from_segments[1:]:
                avoid_nodes.add(edge[0])
                avoid_nodes.add(edge[1])
            avoid_nodes.discard(start_edge[0])
            avoid_nodes.discard(start_edge[1])
            
            # Find random walk of 2 edges from breakpoint_node
            divert_to_post = self._find_alternate_segment(breakpoint_node, divert_from_segments, avoid_nodes, breakpoint_dist)
            
            if divert_to_post:
                divert_to_segments = divert_to_post
                candidates.append((divert_from_segments[1:], divert_to_segments))

        if not candidates:
            # Fallback if no valid configuration found
            divert_from_edges = []
            if top_edges:
                divert_from_edges = [top_edges[0]]  # Or keep it empty depending on requirements
            divert_to_edges = []
        else:
            # Randomly choose a breakpoint configuration
            selected_candidate = random.choice(candidates)
            divert_from_edges = selected_candidate[0]
            divert_to_edges = selected_candidate[1]

        # Convert paths to objective arrays
        divert_from = np.zeros(self.max_num_edges, dtype=int)
        divert_to = np.zeros(self.max_num_edges, dtype=int)

        for e, edge in enumerate(self.both_edges):
            if edge in divert_from_edges or (edge[1],edge[0]) in divert_from_edges:
                divert_from[e] = 1
            if edge in divert_to_edges or (edge[1],edge[0]) in divert_to_edges:
                divert_to[e] = 1
             # Padded entries remain 0
        return {**base_state, 'divert_from_objective': divert_from, 'divert_to_objective': divert_to}

    def _find_alternate_segment(self, start_node, divert_from_edges, avoid_nodes, breakpoint_dist):
        """Find a random n-segment path starting from start_node avoiding avoid_nodes."""
        avoid_edges = set(divert_from_edges)
        for e in divert_from_edges:
            avoid_edges.add((e[1], e[0]))
            
        def dfs(current_node, current_path, visited_nodes):
            # Base case: Path reached the desired length
            if len(current_path) == self.objective_path_length:
                # Should end at a node strictly less than breakpoint_dist from sink
                if self.node_distances.get(current_node, float('inf')) < breakpoint_dist:
                    return list(current_path)
                return None
                
            if current_node not in self.edge_groups:
                return None
                
            valid_edges = []
            for edge in self.edge_groups[current_node]['out']:
                if edge not in self.all_both_edges:
                    continue
                if edge in avoid_edges:
                    continue
                next_node = edge[1]
                if next_node == start_node or next_node in avoid_nodes or next_node in visited_nodes:
                    continue
                    
                valid_edges.append(edge)
                
            random.shuffle(valid_edges)
            
            for edge in valid_edges:
                next_node = edge[1]
                visited_nodes.add(next_node)
                current_path.append(edge)
                
                result = dfs(next_node, current_path, visited_nodes)
                if result:
                    return result
                    
                # Backtrack
                current_path.pop()
                visited_nodes.remove(next_node)
                
            return None

        return dfs(start_node, [], {start_node})

    def _extract_max_flow_path(self, flows):
        """Extract the path with maximum flow from flows dictionary."""
        from_path = []
        current_node = self.super_sink_nodes[0]
        source = 1

        # Safety guards: track visited nodes and limit steps to avoid infinite loops
        visited = {current_node}
        max_steps = max(2 * len(self.nodes), 1000)
        steps = 0

        while current_node != source and steps < max_steps:
            steps += 1
            incoming_edges = self.edge_groups.get(current_node, {}).get('in', [])
            if not incoming_edges:
                break

            prev_edge = max(incoming_edges, key=lambda e: flows.get(e, 0) + random.random() * 1e-6)
            from_path.append(prev_edge)
            current_node = prev_edge[0]

            if current_node in visited:
                # cycle detected; stop walking
                break
            visited.add(current_node)

        path = list(reversed(from_path))
        # preserve previous slicing behaviour but handle short paths safely
        if len(path) <= 2:
            return path
        return path[1:-1]
    
    def step(self, action):                                                     
        """Execute one step in the environment based on the given action."""
        # Initialize step variables
        remaining_budget = self.state['budget'].copy()

        # Determine if action was "do nothing"
        if action == self.max_num_edges:
            remaining_budget[0] = 0

        # Validate action
        action_mask = self.mask_fn()
        
        # Check if action is valid: Must be within actual edges (not padding) and Must be allowed by mask_fn
        valid_action = (action < self.num_both_edges) and (action_mask[action] == 1)
        
        if valid_action:
            # Mark edge as interdicted
            self.state['edge_interdicted'][action] += 1

        # Deduct cost from budget
        action_cost = self.state['edge_costs'][action] if action < self.num_both_edges else 0
        remaining_budget[0] = max(0, remaining_budget[0] - action_cost)
        
        # Check if episode is complete (Determines if org_reward or potential goes to 0)
        done = self._is_episode_complete(remaining_budget)

        if done:
            current_obj = self._evaluate_network_for_strategy(for_potential=False)
            self._cache_flow_array()
            orig_reward = self._calculate_original_reward(current_obj)
            next_potential = 0.0
        elif valid_action:
            current_obj = self._evaluate_network_for_strategy(for_potential=True)
            self._cache_flow_array()
            orig_reward = 0.0
            next_potential = self._calculate_potential(current_obj) 
        else:
            orig_reward = 0.0
            next_potential = self.current_potential  # No change in potential if action is invalid
                                    
        # Apply Potential-Based Reward Shaping (PBRS)
        gamma = 0.999
        potential_reward = (gamma * next_potential) - self.current_potential
                
        # 6. Apply Cost Penalty and Scale
        unscaled_reward = orig_reward + potential_reward - (0.01 * action_cost)
        reward = unscaled_reward / self.reference_budget
            
        # 7. Update Trackers for next step
        self.current_potential = next_potential
        self.last_obj = max(orig_reward, next_potential)

        # Update state
        self.state['budget'] = remaining_budget
    
        return self.state, float(reward), bool(done), False, {}

    def _evaluate_network_for_strategy(self, for_potential=False):
        """Evaluates the network once per step based on strategy, returning the raw objective."""
        if self.attacker_strategy == "zero_sum":
            obj, self.reference_flows = self._compute_objective_and_flows() #No difference for potential vs original
        elif self.attacker_strategy == "canalize":
            obj, self.reference_flows = self._calculate_canalize_objective_and_flows(for_potential=for_potential)
        elif self.attacker_strategy == "isolate":
            obj, self.reference_flows = self._calculate_isolate_objective_and_flows() #No difference for potential vs original
        elif self.attacker_strategy == "divert":
            obj, self.reference_flows = self._calculate_divert_objective_and_flows(for_potential=for_potential)
        else:
            obj, self.reference_flows = 0, {}
        return obj
    
    def _calculate_original_reward(self, current_obj):
        """Calculates the original reward for the current state."""
        if self.attacker_strategy in ["zero_sum", "isolate"]:
            return self.reference_obj - current_obj
        elif self.attacker_strategy in ["canalize", "divert"]:
            return current_obj - self.reference_obj
        return current_obj
    
    def _calculate_potential(self, current_obj):
        """Calculates the potential Phi(s) of the current state."""
        if self.attacker_strategy in ["zero_sum", "isolate"]:
            return self.reference_obj - current_obj
        elif self.attacker_strategy in ["canalize", "divert"]:
            return current_obj
        return current_obj

    def _flows_dict_to_array(self, flows_dict):
        """Convert a flow dictionary to a compact float32 array aligned with self.both_edges."""
        if not flows_dict:
            return np.zeros((self.num_both_edges, 2), dtype=np.float32)
        
        # Use cached flow array logic logic explicitly here to create standard serialized format
        # Forward flow on column 0, reverse flow on column 1
        return np.array(
            [[flows_dict.get(e, 0), flows_dict.get(re, 0)] for e, re in zip(self.both_edges, self.reverse_edges_list)], 
            dtype=np.float32
        )

    def _calculate_stochastic_objective_and_flow(self, strategy_type="zero_sum", return_full_flows=False, for_potential=False):
        """
        Optimized stochastic calculation: group by unique outcomes and weight by probability.
        
        Optimized for Serialization: Stores and returns numpy arrays for flows when return_full_flows=True.
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
                outcome_weights[outcome] = float(np.round(prob, 10))
        
            unique_outcomes.sort()
        else:
            # Sample interdiction outcomes
            outcome_samples = []
            for _ in range(total_samples):
                if self.multiple_interdiction_attempts:
                    failure_probs = ((1 - probs) ** interdicted)
                    success = np.random.binomial(1, 1-failure_probs)
                    outcome = tuple(np.minimum(interdicted, success)) 
                else:
                    success = np.random.binomial(1, probs)
                    outcome = tuple(np.minimum(interdicted, success))
            
                outcome_samples.append(outcome)
        
            outcome_counts = Counter(outcome_samples)
            unique_outcomes = sorted(list(outcome_counts.keys()))
            outcome_weights = {outcome: count / total_samples for outcome, count in outcome_counts.items()}

        # Mode-aware cache key to avoid objective leakage across evaluation modes.
        cache_scope = bool(for_potential)

        def _cache_key(outcome):
            return (cache_scope, outcome)

        # --- MEMOIZATION START ---
        outcomes_needed_from_central = []
        
        # 1. Check Local Cache
        if self.enable_outcome_caching:
            for outcome in unique_outcomes:
                outcome_key = _cache_key(outcome)
                is_valid_hit = outcome_key in self.local_outcome_cache
                # Check for cached flow array OR basic indices
                if is_valid_hit and return_full_flows:
                    cached_item = self.local_outcome_cache[outcome_key]
                    if 'flow_array' not in cached_item and 'flows' not in cached_item:
                         is_valid_hit = False
                    # Ensure cache hit has core flows if feature is active
                    if getattr(self, 'core_flow_extractor', False) and 'core_flow_array' not in cached_item:
                         is_valid_hit = False
                
                if not is_valid_hit:
                    outcomes_needed_from_central.append(outcome)
        else:
            outcomes_needed_from_central = list(unique_outcomes)
        
        # 2. Check Central Cache (only for what wasn't in local)
        outcomes_to_solve = []
        if self.enable_outcome_caching and outcomes_needed_from_central and self.outcome_memo_actors:
            import zlib
            num_shards = len(self.outcome_memo_actors)
            
            shard_keys = defaultdict(list)
            shard_outcomes = defaultdict(list)
            for outcome in outcomes_needed_from_central:
                outcome_key = _cache_key(outcome)
                shard_idx = zlib.adler32(str(outcome_key).encode()) % num_shards
                shard_keys[shard_idx].append(outcome_key)
                shard_outcomes[shard_idx].append(outcome)
            
            futures = []
            shard_indices = []
            for idx, keys in shard_keys.items():
                futures.append(self.outcome_memo_actors[idx].get_batch.remote(keys))
                shard_indices.append(idx)
            
            if futures:
                all_results = ray.get(futures)
                
                for i, results in enumerate(all_results):
                    shard_idx = shard_indices[i]
                    outcomes = shard_outcomes[shard_idx]
                    keys = shard_keys[shard_idx]
                    for outcome, outcome_key, res in zip(outcomes, keys, results):
                        is_valid_result = res is not None
                        if is_valid_result and return_full_flows and 'flow_array' not in res and 'flows' not in res:
                            is_valid_result = False
                            
                        if is_valid_result:
                            self.local_outcome_cache[outcome_key] = res 
                        else:
                            outcomes_to_solve.append(outcome)
        else:
            outcomes_to_solve = outcomes_needed_from_central
        # --- MEMOIZATION END ---

        # 3. Solve Max Flow for truly missing outcomes
        new_results_for_central = {}
        if self.enable_outcome_caching:
            working_cache = self.local_outcome_cache
        else:
            working_cache = {}

        for outcome in outcomes_to_solve:
            outcome_key = _cache_key(outcome)
            # Convert outcome to capacity dict
            capacity_dict = {}
            for idx, edge in enumerate(self.both_edges):
                base_capacity = self.state['edge_capacity'][idx]
                is_interdicted = outcome[idx]
                capacity_dict[edge] = 0 if is_interdicted else base_capacity
        
            # Solve max flow
            obj_val, flows = self.solve_max_flow(capacity_dict, routing_assumption=strategy_type)
        
            objective = self._compute_raw_objective(strategy_type, flows, for_potential, obj_val)

            res = {
                'objective': objective
            }
            
            if return_full_flows:
                # OPTIMIZATION: Convert dict to array immediately for storage
                res['flow_array'] = self._flows_dict_to_array(flows)
                if getattr(self, 'core_flow_extractor', False):  
                    res['core_flow_array'] = getattr(self, '_current_core_flow_array', np.zeros(self.num_both_edges, dtype=np.float32))
            else:
                if self.state['budget'][0] < self.min_edge_cost:
                    res['nonzero_flow_indices'] = []
                else:
                    indices = []
                    for edge, flow in flows.items():
                        if flow > 0:
                            if edge in self.edge_to_index:
                                indices.append(self.edge_to_index[edge])
                            elif (edge[1], edge[0]) in self.edge_to_index:
                                indices.append(self.edge_to_index[(edge[1], edge[0])])
                    res['nonzero_flow_indices'] = sorted(list(set(indices)))
            
            working_cache[outcome_key] = res 
            new_results_for_central[outcome_key] = res
            
        # 4. Update Central Cache (Async)
        if self.enable_outcome_caching and new_results_for_central and self.outcome_memo_actors:
            import zlib
            num_shards = len(self.outcome_memo_actors)
            
            shard_updates = defaultdict(lambda: ([], []))
            
            for outcome_key, res in new_results_for_central.items():
                shard_idx = zlib.adler32(str(outcome_key).encode()) % num_shards
                keys, values = shard_updates[shard_idx]
                keys.append(outcome_key)
                values.append(res)
            
            for idx, (keys, vals) in shard_updates.items():
                self.outcome_memo_actors[idx].set_batch.remote(keys, vals)

        # 5. Compute weighted averages
        ordered_outcomes = list(unique_outcomes)
        weights = np.array([outcome_weights[o] for o in ordered_outcomes])
        objectives = np.array([working_cache[_cache_key(o)]['objective'] for o in ordered_outcomes])
        weighted_objective = np.dot(objectives, weights)
        
        # If not requesting flows, we are done
        if not return_full_flows:
             return weighted_objective, {}

        # Optimized Vectorized Flow Accumulation
        final_flow_array = np.zeros((self.num_both_edges, 2), dtype=np.float32)
        
        # Fast path: All items have flow_array, Efficient stacking
        all_flow_arrays = np.stack([working_cache[_cache_key(o)]['flow_array'] for o in ordered_outcomes])
        final_flow_array = np.average(all_flow_arrays, weights=weights, axis=0)
        
        # ---> ADD THIS: Stack and average the core flows!
        if getattr(self, 'core_flow_extractor', False):
            if all('core_flow_array' in working_cache[_cache_key(o)] for o in ordered_outcomes):
                all_core_arrays = np.stack([working_cache[_cache_key(o)]['core_flow_array'] for o in ordered_outcomes])
                self.core_flows = np.average(all_core_arrays, weights=weights, axis=0)
            else:
                self.core_flows = np.zeros(self.num_both_edges, dtype=np.float32)

        # Return array directly instead of dict
        return weighted_objective, final_flow_array
    
    def _compute_raw_objective(self, strategy_type, flows, for_potential=False, max_flow_obj=0):
        """Maps a flow dictionary to a scalar objective based on the strategy."""
        if strategy_type == "zero_sum":
            return max_flow_obj
        elif strategy_type == "canalize":
            if not for_potential:
                return self._calculate_target_path_flow(flows, 'canalize_objective')
            return self._calculate_canalize_potential(flows)
        elif strategy_type == "isolate":
            return self._calculate_target_edge_flow(flows, 'isolate_objective')
        elif strategy_type == "divert":
            if not for_potential:
                from_flow = self._calculate_target_path_flow(flows, 'divert_from_objective')
                to_flow = self._calculate_target_path_flow(flows, 'divert_to_objective')
                ref_flows = getattr(self, 'reference_start_flows', None)
                ref_a, ref_b = (ref_flows[0], ref_flows[1]) if ref_flows else (0, 0)
                return min((ref_a - from_flow), (to_flow - ref_b))
            return self._calculate_divert_potential(flows)
        return 0.0

    def _compute_objective_and_flows(self, deterministic_mode=None):
        """Calculate the max flow objective and edge flows."""
        if deterministic_mode is None:
            deterministic_mode = self.deterministic_outcomes
        
        if deterministic_mode:
            objective, flows = self.solve_max_flow()
            if getattr(self, 'core_flow_extractor', False):
                self.core_flows = getattr(self, '_current_core_flow_array', np.zeros(self.num_both_edges, dtype=np.float32))
            return objective, self._flows_dict_to_array(flows)
        else:
            objective, flows_array = self._calculate_stochastic_objective_and_flow('zero_sum', return_full_flows=True)
            return objective, flows_array

    def _calculate_canalize_objective_and_flows(self, for_potential=False):
        """Calculate objective for canalize strategy (flow through specific path)."""
        if self.deterministic_outcomes:
            _, flows = self.solve_max_flow(routing_assumption='canalize')
            obj = self._compute_raw_objective('canalize', flows, for_potential)
            if getattr(self, 'core_flow_extractor', False):
                self.core_flows = getattr(self, '_current_core_flow_array', np.zeros(self.num_both_edges, dtype=np.float32))
            return obj, self._flows_dict_to_array(flows)
        return self._calculate_stochastic_objective_and_flow('canalize', return_full_flows=True, for_potential=for_potential)
        
    def _calculate_isolate_objective_and_flows(self):
        """Calculate objective for isolate strategy (reduce flow on specific edges)."""
        if self.deterministic_outcomes:
            _, flows = self.solve_max_flow(routing_assumption='isolate')
            obj = self._compute_raw_objective('isolate', flows)
            if getattr(self, 'core_flow_extractor', False):
                self.core_flows = getattr(self, '_current_core_flow_array', np.zeros(self.num_both_edges, dtype=np.float32))
            return obj, self._flows_dict_to_array(flows)
        return self._calculate_stochastic_objective_and_flow('isolate', return_full_flows=True)

    def _calculate_divert_objective_and_flows(self, mode=None, for_potential=False):
        """Calculate objective for divert strategy (redirect flow from one path to another)."""
        mode = mode if mode is not None else self.deterministic_outcomes
        if mode:
            _, flows = self.solve_max_flow(routing_assumption='divert')
            obj = self._compute_raw_objective('divert', flows, for_potential)
            if getattr(self, 'core_flow_extractor', False):
                self.core_flows = getattr(self, '_current_core_flow_array', np.zeros(self.num_both_edges, dtype=np.float32))
            return obj, self._flows_dict_to_array(flows)
        return self._calculate_stochastic_objective_and_flow('divert', return_full_flows=True, for_potential=for_potential)

    def _calculate_target_path_flow_from_array(self, flow_array, objective_key):
        """Calculate total flow through edges marked in the objective using a flow array."""
        path_edges = self._get_objective_path_edges(objective_key)

        if not path_edges:
            return 0.0
    
        target_flows = []
        for edge in path_edges:
            idx = self.edge_to_index.get(edge)
            if idx is not None:
                target_flows.append(flow_array[idx, 0])
            else:
                idx = self.edge_to_index.get((edge[1], edge[0]))
                if idx is not None:
                    target_flows.append(flow_array[idx, 1])
        
        if not target_flows:
            return 0.0
    
        return min(target_flows)

    def _calculate_target_path_flow(self, flows, objective_key):
        """Calculate total flow through edges marked in the objective."""
        path_edges = self._get_objective_path_edges(objective_key)

        if not path_edges:
            return 0.0
    
        target_flows = [flows.get(edge, 0) for edge in path_edges]
    
        # Return minimum flow among target edges (bottleneck)
        return min(target_flows)

    def _calculate_target_edge_flow(self, flows, objective_key):
        """Calculate total flow on edges marked in the objective."""
        objective = self.state[objective_key]
        # Get indices where objective is 1
        target_indices = np.where(objective[:self.num_both_edges] == 1)[0]
        
        target_edges = [self.both_edges[i] for i in target_indices]
    
        # Batch get flows
        flows_array = np.array([(flows.get(edge, 0) + flows.get((edge[1], edge[0]), 0)) for edge in target_edges])
    
        # Return sum flow among target nodes
        return np.sum(flows_array) #np.sum(total_flow)
    
    def _calculate_max_objective_potential(self, objective_key, reference_start_flow):
        """
        Computes the maximum potential objective delta based on the bottleneck capacity
        of the target edges minus the reference starting flow.
        """
        objective_mask = self.state[objective_key][:self.num_both_edges] == 1
        
        if np.any(objective_mask):
            bottleneck_capacity = np.min(self.state['edge_capacity'][:self.num_both_edges][objective_mask])
        else:
            bottleneck_capacity = 0
            
        return bottleneck_capacity - reference_start_flow
    
    def _calculate_canalize_potential(self, flows):
        """
        Computes the potential of the canalize strategy by calculating the average 
        capped flow difference along the canalize path.
        """
        path_edges = self._get_objective_path_edges('canalize_objective')
        
        if not path_edges:
            return 0.0

        flow_diffs = []
        reference_start_flow = getattr(self, 'reference_start_flow', 0) or 0
        max_canalize_objective = getattr(self, 'max_canalize_objective', 0) or 0
        
        for edge in path_edges:
            f_after = flows.get(edge, 0)
            flow_diffs.append(min(f_after - reference_start_flow, max_canalize_objective))
                
        return sum(flow_diffs) / len(flow_diffs)
    
    def _calculate_divert_potential(self, flows):
        """
        Computes the potential of the divert strategy.
        Term 1: Average capped flow increase along the divert_to path.
        Term 2: Capped bottleneck flow reduction on the divert_from path.
        Returns the minimum of these two terms to enforce a strict flow transfer.
        """
        path_edges_to = self._get_objective_path_edges('divert_to_objective')
        
        if not path_edges_to or getattr(self, 'reference_start_flows', None) is None:
            return 0.0

        ref_from, ref_to = self.reference_start_flows
        max_divert_obj = getattr(self, 'max_divert_objective', 0)

        # Term 1: Canalize-style average increase on the 'to' path
        to_flow_diffs = []
        for edge in path_edges_to:
            f_after = flows.get(edge, 0)
            to_flow_diffs.append(min(f_after - ref_to, max_divert_obj))
            
        term1 = sum(to_flow_diffs) / len(to_flow_diffs) if to_flow_diffs else 0.0

        # Term 2: Reduction of bottleneck flow on the 'from' path
        from_bottleneck = self._calculate_target_path_flow(flows, 'divert_from_objective')
        term2 = min(max_divert_obj, ref_from - from_bottleneck)

        # The potential is strictly the amount of flow successfully transferred
        return min(term1, term2)

    def _is_episode_complete(self, remaining_budget, current_obj=None):
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
            if current_obj is not None:
                if current_obj == 0 and self.attacker_strategy in ["zero_sum", "isolate"]:
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

    def load_network_from_state(self, seed, state):
        """Reset the environment to initial state and return observation."""
        # Clean up any existing models
        self._cleanup_models()
        self.strategy_objectives_setup = False # Force objective reset on next solve
        self.old_routing_assumption = False
        self.reference_start_flows = None
        self.reference_start_flow = None
        self.reference_obj = 0.0
        self.reference_budget = 0.0

        # Clear local outcome cache when loading new state
        self.local_outcome_cache = {}
        self._clear_objective_path_cache(('canalize_objective', 'divert_from_objective', 'divert_to_objective'))

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

        # We must set this early before it gets used via mask_fn or solve_max_flow
        num_actions = getattr(self.action_space, 'n', self.num_both_edges)
        self.num_interdictable = min(self.num_both_edges, np.array(num_actions).item() if not isinstance(num_actions, int) else num_actions)

        if self.attacker_strategy in ['canalize', 'divert']:
            self.solve_max_flow(routing_assumption='zero_sum')

            if self.attacker_strategy == 'canalize':
                self._cache_objective_path('canalize_objective')
            elif self.attacker_strategy == 'divert':
                self._cache_objective_paths(('divert_from_objective', 'divert_to_objective'))

        # Calculate reference objective value for the attacker's strategy
        self._initialize_strategy_references(is_reset_loop=False)
        self.reference_budget = state['budget'][0]

        return self.state, {}

    def get_edges_on_paths_to_source(self, start_nodes):
        """
        Determine which edges have non-zero flow in self.cached_flow_array and lie along a path from the isolate_objective nodes to a source node. Uses backward BFS from target nodes through edges with non-zero flow to reach source nodes.
        Returns:
            np.ndarray: Boolean array of shape (self.numbothedges,) where True indicates the edge has non-zero flow and is on a path from an isolate_objective node to a source.
        """
        target_nodes = set()
        for idx in np.where(start_nodes[:self.num_both_edges]==1)[0]:
            edge = self.both_edges[idx]
            target_nodes.add(edge[1])
        
        visited_nodes = set(target_nodes)
        incoming_edge_indices = set()
        
        # Precompute the boolean map of edges with flow for O(1) lookup
        if getattr(self, 'cached_flow_array', None) is not None:
            has_flow_array = self.cached_flow_array[:self.num_both_edges] > 1e-6
        else:
            has_flow_array = np.zeros((self.num_both_edges, 2), dtype=bool)

        # Standard queue-based BFS using the pre-computed edge_groups adjacency dictionary
        queue = list(target_nodes)
        
        while queue:
            curr_node = queue.pop(0)
            
            # Check incoming edges to the current node
            if curr_node in self.edge_groups:
                for edge in self.edge_groups[curr_node]['in']:
                    edge_idx = self.edge_to_index.get(edge)
                    if edge_idx is not None:
                        # Only traverse if the edge actually has forward flow (since edge is in both_edges)
                        if has_flow_array[edge_idx, 0]:
                            incoming_edge_indices.add(edge_idx)
                            prev_node = edge[0]
                            if prev_node not in visited_nodes:
                                visited_nodes.add(prev_node)
                                queue.append(prev_node)
                    else:
                        rev_edge_idx = self.edge_to_index.get((edge[1], edge[0]))
                        if rev_edge_idx is not None:
                            # Edge is reverse of both_edges, flow from prev->curr is reverse flow
                            if has_flow_array[rev_edge_idx, 1]:
                                incoming_edge_indices.add(rev_edge_idx)
                                prev_node = edge[0]
                                if prev_node not in visited_nodes:
                                    visited_nodes.add(prev_node)
                                    queue.append(prev_node)
    
        action_mask = np.zeros(self.num_both_edges, dtype=bool)
        if incoming_edge_indices:
            action_mask[list(incoming_edge_indices)] = True
        return action_mask

    def mask_fn(self):
        """
        Fully vectorized function using cached flow information.
        Separates masking into Resource-Based, Flow-Based, and Mission-Specific stages.
        """
        remaining_budget = self.state['budget'][0]
        edge_interdicted = self.state['edge_interdicted']
    
        action_mask = np.ones(self.action_space.n, dtype=np.float32)
    
        # --- Stage 1: Resource-Based Masking ---
        sufficient_budget = (remaining_budget - self.state['edge_costs'][:self.num_interdictable]) >= -0.1
        has_probability = self.state['edge_interdiction_probability'][:self.num_interdictable] > 0
        within_limit = (edge_interdicted[:self.num_interdictable] + 1) <= self.max_interdictions
        
        resource_mask = sufficient_budget & has_probability & within_limit
        
        # --- Stage 2: Flow-Based Masking ---
        if self.enable_flow_masking and hasattr(self, 'cached_flow_array'):
            has_flow = self.cached_flow_array[:self.num_interdictable].sum(axis=1) > 0
            flow_mask = resource_mask & has_flow
        else:
            flow_mask = resource_mask.copy()
            
        # --- Stage 3: Mission-Specific Masking ---
        if self.enable_mission_masking:
            if self.attacker_strategy == 'isolate':
                # Get edges on paths from isolate objectives to sources
                on_path_to_source = self.get_edges_on_paths_to_source(start_nodes=self.state['isolate_objective'])
                mission_mask = flow_mask & on_path_to_source[:self.num_interdictable]
            elif self.attacker_strategy == 'canalize':
                not_target = self.state['canalize_objective'][:self.num_interdictable] != 1
                mission_mask = flow_mask & not_target
            elif self.attacker_strategy == 'divert':
                not_target = self.state['divert_to_objective'][:self.num_interdictable] != 1
                mission_mask = flow_mask & not_target
            else:
                mission_mask = flow_mask.copy()
        else:
            mission_mask = flow_mask.copy()
            
        # --- Collect Statistics ---
        total_candidates = self.num_interdictable
        res_valid = resource_mask.sum()
        flow_valid = flow_mask.sum()
        mission_valid = mission_mask.sum()
        
        self.last_mask_stats = {
            'resource': int(total_candidates - res_valid),
            'flow': int(res_valid - flow_valid),
            'mission': int(flow_valid - mission_valid)
        }
    
        action_mask[:self.num_interdictable] = mission_mask.astype(np.float32)
    
        return action_mask