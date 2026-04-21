import os
import copy
import logging
import random
import math
import collections
import itertools
from collections import defaultdict, deque
import pickle

import numpy as np
import gurobipy as grb
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import ray
import threading
import time
import zlib

# --- Ray Actors (Moved from env_TA.py) ---

@ray.remote
class _ProgressActor:
    def __init__(self):
        self.count = 0
        self.pruned_count = 0
        self.invalid_count = 0
        self.memo_count = 0
        self.base_count = 0
    def increment(self, n: int = 1, pruned: int = 0, invalid: int = 0, memo: int = 0, base: int = 0):
        self.count += int(n)
        self.pruned_count += int(pruned)
        self.invalid_count += int(invalid)
        self.memo_count += int(memo)
        self.base_count += int(base)
    def get_count(self):
        return self.count, self.pruned_count, self.invalid_count, self.memo_count, self.base_count
    def reset(self):
        self.count = 0
        self.pruned_count = 0
        self.invalid_count = 0
        self.memo_count = 0
        self.base_count = 0

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
    def __init__(self, initial_alpha=-float('inf'), initial_sequence=None):
        self.alpha = initial_alpha
        self.sequence = initial_sequence if initial_sequence is not None else []
    def get(self):
        return self.alpha, self.sequence
    def update(self, new_alpha, new_sequence=None):
        if new_alpha > self.alpha:
            self.alpha = new_alpha
            if new_sequence is not None:
                self.sequence = new_sequence

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
                 max_depth_inner=100, outcome_memo_actor=None, outcome_memo_actors=None, alpha_actor=None,
                 enable_outcome_caching=True, enable_alpha_pruning=True, sample_size=1000, reduce_flow=False, jitter=False, projection_uses_flow=False, env_references=None):
        """
        Worker now accepts a progress_actor handle, a shared memo_actor handle,
        and budget_levels so it can estimate progress for invalid actions.
        """
        import importlib, copy, numpy as np, ray as _ray, time
        from collections import defaultdict
        # Dynamic import to avoid circular dependency at module level
        env_mod = importlib.import_module("env_TA")
        CustomEnv = getattr(env_mod, "CustomEnv")

        # Instantiate env with the same key flags so internals (num_both_edges, spaces, models) are set up
        # Pass None for actors initially to avoid clearing them during load_network_from_state
        self.env = CustomEnv(nodes, edges,
                             deterministic_agent=deterministic_outcomes,
                             multiple_interdiction_attempts=multiple_interdiction_attempts,
                             attacker_strategy=attacker_strategy,
                             outcome_memo_actor=None,
                             outcome_memo_actors=None,
                             sample_size=sample_size)
         
        # Set config flag
        self.env.enable_outcome_caching = enable_outcome_caching
        self.env.reduce_flow = reduce_flow
        self.jitter = jitter
        self.projection_uses_flow = projection_uses_flow

        # Optimize serialization: Avoid deep copying the entire state.
        # We only copy the dynamic parts that change during the episode/search.
        # This allows large static arrays (topology, capacity) to remain read-only/zero-copy.
        self.state = {}
        # Keys for dynamic state components that must be writable
        dynamic_keys = {'budget', 'edge_interdicted'}

        for k, v in state_snapshot.items():
            if k in dynamic_keys:
                # Create writable copy for dynamic elements
                if isinstance(v, np.ndarray):
                    self.state[k] = v.copy()
                elif hasattr(v, 'copy'):
                    self.state[k] = v.copy()
                else:
                    self.state[k] = copy.deepcopy(v)
            else:
                # Zero-copy read-only reference for static elements
                self.state[k] = v

        # Restore state on the worker
        self.env.load_network_from_state(seed, self.state)

        if env_references:
            for k, v in env_references.items():
                setattr(self.env, k, v)

        # Attach shared actors after loading state
        self.env.outcome_memo_actor = outcome_memo_actor
        self.env.outcome_memo_actors = outcome_memo_actors

        self.attacker_strategy = attacker_strategy
        self.min_edge_cost = min_edge_cost
        self.num_both_edges = num_both_edges
        self.max_depth_inner = max_depth_inner
        self.progress_actor = progress_actor
        self.alpha_actor = alpha_actor
        self.enable_alpha_pruning = enable_alpha_pruning
        
        # Handle Sharded Memo Actors
        self.memo_actors = memo_actors
        self.num_memo_shards = len(memo_actors) if memo_actors else 0
        
        self.progress_granularity = int(progress_granularity)
        self.budget_levels = int(budget_levels)
        
        # Initialize persistent local memoization cache (persist across tasks)
        self.memo_local = {} 

        # PRE-COMPUTE HEURISTIC BASE VALUES (Optimization)
        # This avoids multiplying capacity * probability at every node
        self.env.heuristic_base_values = (
            self.env.state['edge_capacity'][:self.num_both_edges] * 
            self.env.state['edge_interdiction_probability'][:self.num_both_edges]
        )

    def evaluate_subtree(self, remaining_budget, interdicted_state, depth, objective_tolerance=1e-5):
        import numpy as np, ray as _ray, time, zlib # Added zlib for stable hashing
        # Use persistent local cache (hot between tasks)
        memo_local = self.memo_local 
        local_counter = 0
        local_pruned_counter = 0
        local_invalid_counter = 0
        local_memo_counter = 0
        local_base_counter = 0
        self.objective_tolerance = objective_tolerance
        
        # Local cache of the global alpha value
        local_alpha_cache = -float('inf')
        pending_alpha_ref = None
        
        # Initialize alpha synchronously at the start of the subtree to ensure we have the latest baseline
        if self.alpha_actor:
            try:
                local_alpha_cache, _ = _ray.get(self.alpha_actor.get.remote())
                # Start the first async fetch for updates
                pending_alpha_ref = self.alpha_actor.get.remote()
            except Exception:
                pass

        #Time-based throttling variables
        last_report_time = time.time()
        report_interval = 0.5  # Max 2 reports per second per worker

        def maybe_flush_progress():
            nonlocal local_counter, local_pruned_counter, local_invalid_counter, local_memo_counter, local_base_counter, last_report_time, local_alpha_cache, pending_alpha_ref
            # 1. Check if we have enough accumulated progress (batch size)
            if self.progress_actor is not None and (local_counter >= self.progress_granularity or local_pruned_counter >= self.progress_granularity or local_invalid_counter >= self.progress_granularity or local_memo_counter >= self.progress_granularity or local_base_counter >= self.progress_granularity):
                # 2. Check if enough time has passed (throttle)
                now = time.time()
                if (now - last_report_time) > report_interval:
                    try:
                        # report and reset local counter (best-effort)
                        self.progress_actor.increment.remote(local_counter, local_pruned_counter, local_invalid_counter, local_memo_counter, local_base_counter)
                        local_counter = 0
                        local_pruned_counter = 0
                        local_invalid_counter = 0
                        local_memo_counter = 0
                        local_base_counter = 0
                        last_report_time = now
                        
                        # Sync alpha asynchronously (NON-BLOCKING)
                        if self.alpha_actor and pending_alpha_ref:
                            # Check if the future is ready
                            ready, _ = _ray.wait([pending_alpha_ref], timeout=0)
                            if ready:
                                try:
                                    remote_val, _ = _ray.get(pending_alpha_ref)
                                    if remote_val > local_alpha_cache:
                                        local_alpha_cache = remote_val
                                except Exception:
                                    pass
                                # Launch next request
                                pending_alpha_ref = self.alpha_actor.get.remote()
                    except Exception:
                        pass

        # INITIALIZE STATE ONCE (Optimization: Avoid copying state at every node)
        self.env.state['budget'][0] = remaining_budget
        self.env.state['edge_interdicted'][:] = interdicted_state

        def dp_local(d, alpha=-float('inf')):
            nonlocal local_counter, local_pruned_counter, local_invalid_counter, local_memo_counter, local_base_counter, local_alpha_cache
            
            # Read current state directly from env (already synchronized)
            # Use slice for key generation
            current_interdicted = self.env.state['edge_interdicted']
            key = current_interdicted[:self.num_both_edges].tobytes()
            rem_budget = self.env.state['budget'][0]
            
            # Incorporate global knowledge
            alpha = max(alpha, local_alpha_cache)

            # 1) check local cache
            if key in memo_local:
                # ADDED: Count volume for cached hit
                vol = int(int(self.num_both_edges) ** max(0, self.budget_levels - d))
                local_counter += vol
                local_memo_counter += vol
                maybe_flush_progress()
                return memo_local[key]

            # 2) check centralized memo (SHARDED)
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
                local_memo_counter += vol
                maybe_flush_progress()
                return shared_val

            # NO STATE SAVE/RESTORE NEEDED HERE - State is already set

            # terminal objective
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
                memo_local[key] = (final_objective, [])
                # Count the volume of the skipped subtree (leaves)
                volume = int(int(self.num_both_edges) ** max(0, self.budget_levels - d))
                local_counter += volume
                local_base_counter += volume
                maybe_flush_progress()
                # publish to central memo (best-effort, async)
                if target_actor is not None:
                    try:
                        target_actor.set.remote(key, memo_local[key])
                    except Exception:
                        pass
                return final_objective, []

            action_mask = self.env.mask_fn()

            valid_actions = np.where(action_mask[:self.num_both_edges] == 1)[0]

            # Report the discovery of invalid actions as progress using estimated subtree sizes
            num_invalid = int(self.num_both_edges) - len(valid_actions)
            if num_invalid > 0 and self.progress_actor is not None:
                try:
                    # estimated states pruned by each invalid action
                    est_per_invalid = int(int(self.num_both_edges) ** max(0, self.budget_levels - (d + 1)))
                    invalid_vol = num_invalid * est_per_invalid
                    local_counter += invalid_vol
                    local_invalid_counter += invalid_vol
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

            if self.enable_alpha_pruning:
                # Heuristic sorting for pruning
                # Optimization: State is already set correctly, just call heuristics
                
                # IMPORTANT: For zero_sum, we need to handle negative heuristics appropriately if calculate_action_heuristics returns positives
                heuristics = self.env.calculate_action_heuristics(valid_actions, current_flows, rem_budget, jitter=getattr(self, 'jitter', False), projection_uses_flow=self.projection_uses_flow)
                
                # Optimization: Mass-prune actions that can't beat alpha using vectorized boolean masking
                # This avoids expensive sorting (N log N) for actions that will be immediately pruned anyway
                if alpha > -1e9: 
                     # Calculate pruning mask for all actions at once
                     # Condition: Is valid_upper_bound < best_known_alpha?
                     keep_mask = (final_objective + heuristics) >= (alpha - self.objective_tolerance)
                     n_pruned = len(valid_actions) - np.count_nonzero(keep_mask)
                     
                     if n_pruned > 0:
                         # 1. Update progress for the entire batch of pruned subtrees
                         est_per_skipped = int(int(self.num_both_edges) ** max(0, self.budget_levels - (d + 1)))
                         pruned_vol = n_pruned * est_per_skipped
                         local_counter += pruned_vol
                         local_pruned_counter += pruned_vol
                         maybe_flush_progress()
                         
                         # 2. Apply filter to reduce array sizes
                         valid_actions = valid_actions[keep_mask]
                         heuristics = heuristics[keep_mask]
                         
                         # Early exit if everything was pruned
                         if len(valid_actions) == 0:
                             return best_reward, best_seq

                # Check if we should reverse sort order based on strategy sign convention
                # calculate_action_heuristics usually returns positive "impact" (flow reduction)
                # But our objective here (final_objective) might be negative.
                
                # Sort descending
                sorted_indices = np.argsort(-heuristics)
                valid_actions = valid_actions[sorted_indices]
                heuristics = heuristics[sorted_indices]
            
            for i, action in enumerate(valid_actions):
                # Pruning
                if self.enable_alpha_pruning:
                     # Pruning condition using the new consolidated heuristic
                     if final_objective + heuristics[i] < alpha - self.objective_tolerance:
                         skipped_actions = len(valid_actions) - i
                         est_per_skipped = int(self.num_both_edges ** max(0, self.budget_levels - (d + 1)))
                         pruned_vol = skipped_actions * est_per_skipped
                         local_counter += pruned_vol
                         local_pruned_counter += pruned_vol
                         maybe_flush_progress()
                         # Do not memoize results from pruned nodes as they may be valid for lower alphas
                         return best_reward, best_seq

                # PROPOSED OPTIMIZATION: Incremental Update
                # No copying, just modify in place
                self.env.state['edge_interdicted'][action] += 1
                cost = self.env.state['edge_costs'][action]
                self.env.state['budget'][0] -= cost
                
                # Recurse
                fut_reward, fut_seq = dp_local(d + 1, alpha)
                
                # Backtrack (Incremental Revert)
                self.env.state['edge_interdicted'][action] -= 1
                self.env.state['budget'][0] += cost
                
                if fut_reward > best_reward + self.objective_tolerance:
                    best_reward = fut_reward
                    best_seq = [action] + fut_seq
                    alpha = max(alpha, best_reward)
                    
                    # Update global alpha if we found something better
                    if alpha > local_alpha_cache:
                        local_alpha_cache = alpha
                        if self.alpha_actor:
                            try:
                                self.alpha_actor.update.remote(alpha)
                            except: pass
                elif abs(fut_reward - best_reward) <= self.objective_tolerance:
                    # Tie-breaking within tolerance: favor sequence with the minimal total cost (highest remaining depth)
                    current_cost = sum(self.env.state['edge_costs'][a] for a in best_seq)
                    new_cost = self.env.state['edge_costs'][action] + sum(self.env.state['edge_costs'][a] for a in fut_seq)
                    if new_cost < current_cost:
                        best_seq = [action] + fut_seq

            memo_local[key] = (best_reward, best_seq)
            
            # publish result to central memo (best-effort, async)
            if target_actor is not None:
                try:
                    target_actor.set.remote(key, memo_local[key])
                except Exception:
                    pass

            return best_reward, best_seq

        result = dp_local(depth)
        # flush any remaining progress
        if self.progress_actor is not None and (local_counter > 0 or local_pruned_counter > 0 or local_invalid_counter > 0 or local_memo_counter > 0 or local_base_counter > 0):
            try:
                self.progress_actor.increment.remote(local_counter, local_pruned_counter, local_invalid_counter, local_memo_counter, local_base_counter)
            except Exception:
                pass
        return result

    def expand_frontier_batch(self, nodes_data):
        """
        Expands a batch of frontier nodes in parallel.
        nodes_data: list of (node_id, budget, state, depth)
        Returns: list of (parent_id, node_val, valid_children_data, is_terminal)
                 where valid_children_data is list of (child_budget, child_state, action)
        """
        results = []
        
        # Optimization: Local references
        env = self.env
        check_budget = env.state['budget']
        check_interdicted = env.state['edge_interdicted']
        
        for n_id, budget, state, depth in nodes_data:
            # 1. Restore State
            check_budget[0] = budget
            check_interdicted[:] = state
            
            # 2. Compute Objective & Flows (Heavy Compute)
            if self.attacker_strategy == "zero_sum":
                val, flows = env._compute_objective_and_flows()
                val = -val 
            elif self.attacker_strategy == 'canalize':
                val, flows = env._calculate_canalize_objective_and_flows()
            elif self.attacker_strategy == 'isolate':
                val, flows = env._calculate_isolate_objective_and_flows()
                val = -val
            elif self.attacker_strategy == 'divert':
                val, flows = env._calculate_divert_objective_and_flows()
            else:
                val = -float('inf')
                flows = {}
            
            # Update reference flows for mask_fn mechanism
            env.reference_flows = flows
            env._cache_flow_array()

            # 3. Get Valid Actions
            action_mask = env.mask_fn()
            valid_actions = np.where(action_mask[:self.num_both_edges] == 1)[0]
            
            # 4. Report Progress for Pruned Branches
            num_invalid = int(self.num_both_edges) - len(valid_actions)
            if num_invalid > 0 and self.progress_actor:
                # Calculate volume of pruned subtrees
                # remaining_depth = budget_levels - (current_depth + 1)
                child_remaining_depth = max(0, self.budget_levels - (depth + 1))
                child_volume = int(self.num_both_edges) ** child_remaining_depth
                inv_vol = num_invalid * child_volume
                self.progress_actor.increment.remote(inv_vol, 0, inv_vol, 0, 0)
            
            # 5. Generate Children Data
            children_data = []
            if len(valid_actions) > 0:
                # Calculate heuristics to sort children (optional but good for efficiency)
                # heuristics = env.calculate_action_heuristics(valid_actions, flows, budget)
                # sorted_indices = np.argsort(-heuristics)
                # valid_actions = valid_actions[sorted_indices]

                for action in valid_actions:
                    # Apply action temporarily involves copying mechanism since we are in a loop
                    # But since we return data, we just calculate new values
                    cost = env.state['edge_costs'][action]
                    new_budget = budget - cost
                    
                    # Create new state array (must copy)
                    new_state = state.copy()
                    new_state[action] += 1
                    
                    children_data.append((new_budget, new_state, action))
            
            is_terminal = (len(children_data) == 0)
            results.append((n_id, val, children_data, is_terminal))
            
        return results

# --- Mixin Class ---

class InterdictionSolverMixin:
    """
    Mixin containing optimization and machine learning solver methods for Network Interdiction.
    """

    def solve_optimal_interdiction(self, method='monolithic', threads=None, time_limit=None):
        if self.deterministic_outcomes == True:
            # Deterministic: Solve using Model 1D
            if not hasattr(self, 'optimal_model'):
                # Initializing the model
                self.optimal_model = grb.Model("Optimal Model", env=self.GUROBI_ENV)

            if threads is not None:
                self.optimal_model.setParam("Threads", threads)
            
            if time_limit is not None:
                self.optimal_model.setParam("TimeLimit", time_limit)
                
            if not hasattr(self, 'gamma'):
                # Creating decision variables
                self.gamma = self.optimal_model.addVars(self.both_edges, vtype=grb.GRB.BINARY, name="gamma")
                self.alpha = self.optimal_model.addVars(self.nodes, vtype=grb.GRB.BINARY, name="alpha")
                self.beta = self.optimal_model.addVars(self.edges_reset, vtype=grb.GRB.BINARY, name="beta")

                self.optimal_model.update()

                # Creating constraints
                self.optimal_model.setAttr("UB", [self.gamma[e] for e in self.noninterdictable_edges],0)

                self.gamma_constr = self.optimal_model.addConstrs((self.gamma[e] <= 1 for e in self.both_edges), name="gamma_constr")

                self.budget_constr = self.optimal_model.addConstr(grb.quicksum(self.edges_episode[e].interdiction_cost * self.gamma[e] for e in self.both_edges) <= self.state['budget'][0], name="budget")

                self.source_sink_constr = self.optimal_model.addConstr(self.alpha[self.super_sink_nodes[0]] - self.alpha[self.super_source_nodes[0]] >= 1, name="source_sink")

                self.aabg_constr = self.optimal_model.addConstrs((self.alpha[e[0]] - self.alpha[e[1]] + self.beta[e] + self.gamma[e] >= 0 for e in self.both_edges), name='aabg')
                self.aabg_reverse_constr = self.optimal_model.addConstrs((self.alpha[e[1]] - self.alpha[e[0]] + self.beta[e] + self.gamma[e] >= 0 for e in self.both_edges), name='aabg')

                self.optimal_model.setObjective(grb.quicksum(self.edges_episode[e].capacity * self.beta[e] for e in self.both_edges), grb.GRB.MINIMIZE)

            self.optimal_model.optimize()

            # Store Results
            interdicted_edges = [e for e in self.both_edges if self.gamma[e].x > 0.5]
            
            return(self.optimal_model.objVal, interdicted_edges)
        
        else:  #Solve Stochastic Case with Cormican's Formulation      
            M = 200                       # Number of training episodes
            N = 700                   # Number of test episodes
            seed_list = [100, 200, 300]#, 400, 500]
            best_objective_value = 100000    # Big M Value
            best_interdicted_edges = []
            unique_interdicted_sets = []
            
            # Determine tasks: standard M for all methods (monolithic and decomposition)
            tasks = [(s, M) for s in seed_list]

            # Test multiple solutions
            for seed, n_scens in tasks:
                if self.multiple_interdiction_attempts:
                    objective_value, interdicted_edges, interdicted_quantities = self.solve_stochastic_max_flow_IM(n_scenarios=n_scens, seed=seed, method=method, threads=threads, time_limit=time_limit)
                    
                    # Create dense vector for key (values > 1 allowed)
                    interdiction_vector = np.zeros(len(self.both_edges), dtype=int)
                    for edge, qty in zip(interdicted_edges, interdicted_quantities):
                        if edge in self.edge_to_index:
                            interdiction_vector[self.edge_to_index[edge]] = qty
                    interdicted_key = tuple(interdiction_vector)
                    
                else:
                    objective_value, interdicted_edges = self.solve_stochastic_max_flow(n_scenarios=n_scens, seed=seed, method=method, threads=threads, time_limit=time_limit)
                    
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
                        objective_value, interdicted_edges, interdicted_quantities = self.solve_stochastic_max_flow_IM(n_scenarios=N, interdicted_edges=interdicted_edges, interdicted_quantities=interdicted_quantities, method=method, threads=threads, time_limit=time_limit)
                        # Expand for return value
                        current_solution = []
                        for e, k in zip(interdicted_edges, interdicted_quantities):
                            current_solution.extend([e] * k)
                    else:
                        # Use fast evaluation for both decomposition and monolithic
                        objective_value = self._evaluate_solution_stochastic(interdicted_edges)
                        current_solution = interdicted_edges

                    if objective_value < best_objective_value:
                        best_objective_value = objective_value
                        best_interdicted_edges = current_solution

            return best_objective_value, best_interdicted_edges

    def _evaluate_solution_stochastic(self, interdicted_edges):
        """
        Evaluate a specific interdiction solution using the environment's 
        stochastic objective calculation. Wraps _calculate_stochastic_objective_and_flow
        handling state transitions.
        """
        # Backup state
        old_budget = self.state['budget'][0]
        old_interdicted = self.state['edge_interdicted'].copy()
        
        # Apply hypothetical strategy
        # Reset interdiction
        self.state['edge_interdicted'] = np.zeros_like(self.state['edge_interdicted'])
        
        # Set new interdictions
        for e in interdicted_edges:
            idx = self.edge_to_index[e]
            # Assuming single attempt/level for this interface: set to 1
            # If multiple attempts are supported, this sets it to 1 attempt.
            self.state['edge_interdicted'][idx] += 1 

        try:
            # Calculate objective (returns expected value)
            objective, _ = self._calculate_stochastic_objective_and_flow(self.attacker_strategy, return_full_flows=False)
        finally:
            # Restore state
            self.state['budget'][0] = old_budget
            self.state['edge_interdicted'] = old_interdicted
            
        return objective

    def _solve_stochastic_decomposition(self, n_scenarios, seed, interdicted_edges, threads=None, time_limit=None):
        np.random.seed(seed)
        
        # 1. Generate Scenarios (Consistent with Monolithic)
        probs = self.state["edge_interdiction_probability"][:self.num_both_edges]
        scenario_outcomes = np.random.binomial(1, probs, size=(n_scenarios, len(self.both_edges)))
        
        # 2. Master Problem
        start_time = time.time()
        master = grb.Model("Master_Benders", env=self.GUROBI_ENV)
        
        if threads is not None:
            master.setParam("Threads", threads)
        if time_limit is not None:
            master.setParam("TimeLimit", time_limit)

        gamma = master.addVars(self.both_edges, vtype=grb.GRB.BINARY, name="gamma")
        theta = master.addVar(lb=0, name="theta")
        
        # Constraints
        # Fixed Interdictions
        if interdicted_edges:
            for e in interdicted_edges:
                 master.addConstr(gamma[e] == 1)
        
        master.setAttr("UB", [gamma[e] for e in self.noninterdictable_edges], 0)

        # Budget
        master.addConstr(grb.quicksum(self.edges_episode[e].interdiction_cost * gamma[e] for e in self.both_edges) <= self.state['budget'][0], name="budget")
        
        master.setObjective(theta, grb.GRB.MINIMIZE)
        
        # 3. Subproblem Prep
        # Use centralized model via solve_max_flow(capacity_dict=...)
        
        # Benders Loop
        LB = -float('inf')
        UB = float('inf')
        epsilon = 1e-4
        max_iter = 100
        
        # Initialize x_hat to avoid reference error if loop breaks early
        x_hat = {}

        for iteration in range(max_iter):
            if time_limit is not None and time.time() - start_time > time_limit:
                break
                
            master.optimize()
            if master.status != grb.GRB.OPTIMAL:
                break
                
            x_hat = {e: gamma[e].X for e in self.both_edges}
            current_theta = theta.X
            LB = current_theta 
            
            # Solve Subproblems
            total_flow = 0.0
            cut_term_coefs = defaultdict(float) # Sum across scenarios of coefs for gamma[e]
            
            for s in range(n_scenarios):
                # Update Capacities
                current_caps = {}
                for idx, e in enumerate(self.both_edges):
                    # Interdiction is successful if attempted (x_hat > 0.5) AND outcome is 1
                    is_blocked = (x_hat[e] > 0.5) and (scenario_outcomes[s, idx] == 1)
                    cap = 0 if is_blocked else self.edges_episode[e].capacity
                    current_caps[e] = cap
                    # Reverse edge
                    rev_e = (e[1], e[0])
                    current_caps[rev_e] = cap
                
                # Solve max flow using environment method
                sub_obj, flow_dict = self.solve_max_flow(capacity_dict=current_caps)
                
                total_flow += sub_obj
                
                # Retrieve flows to build Benders cut
                for idx, e in enumerate(self.both_edges):
                     outcome = scenario_outcomes[s, idx]
                     
                     # Get flow on forward edge e
                     f_fwd = flow_dict.get(e, 0)
                     
                     # Get flow on reverse edge 
                     rev_e = (e[1], e[0])
                     f_rev = flow_dict.get(rev_e, 0)
                     
                     f_total = f_fwd + f_rev
                     
                     # Cut Term: - Flow * Outcome * Gamma
                     cut_term_coefs[e] -= f_total * outcome

            avg_flow = total_flow / n_scenarios
            UB = avg_flow
            
            # Check Convergence
            if avg_flow <= current_theta + epsilon:
                 break

            # Add Cut
            grad_dot_xhat = sum(cut_term_coefs[e] * x_hat[e] for e in self.both_edges)
            intercept = total_flow - grad_dot_xhat
            
            rhs = intercept / n_scenarios
            lhs_terms = grb.quicksum((cut_term_coefs[e]/n_scenarios) * gamma[e] for e in self.both_edges)
            
            master.addConstr(theta >= rhs + lhs_terms)
            master.update()

        # Use x_hat to construct interdicted set, as gamma[e].X may be unavailable if loop
        # terminated via max_iter or optimization failed
        interdicted = [e for e in self.both_edges if x_hat.get(e, 0) > 0.5]
        return UB, interdicted

    def _solve_stochastic_decomposition_IM(self, n_scenarios, seed, interdicted_edges, interdicted_quantities, threads=None, time_limit=None):
        np.random.seed(seed)
        
        # 1. Generate Scenarios (Same as Monolithic IM)
        interdictable_indices = [self.edge_to_index[e] for e in self.interdictable_edges]
        p_base = self.state["edge_interdiction_probability"][interdictable_indices]
        
        k_vals = np.arange(1, self.max_interdictions + 1)
        probs = 1 - (1 - p_base[:, np.newaxis]) ** k_vals
        
        scenario_outcomes = np.random.binomial(1, probs, size=(n_scenarios, len(self.interdictable_edges), len(k_vals)))
        interdictable_edge_map = {e: i for i, e in enumerate(self.interdictable_edges)}
        
        # 2. Master Problem
        start_time = time.time()
        master = grb.Model("Master_Benders_IM", env=self.GUROBI_ENV)

        if threads is not None:
            master.setParam("Threads", threads)
        if time_limit is not None:
            master.setParam("TimeLimit", time_limit)
        
        # Variables: gamma[e, k]
        gamma_indices = [(e, k) for e in self.interdictable_edges for k in range(1, self.max_interdictions + 1)]
        gamma = master.addVars(gamma_indices, vtype=grb.GRB.BINARY, name="gamma")
        theta = master.addVar(lb=0, name="theta")
        
        # Fixed Interdictions
        if interdicted_edges:
            for e, k in zip(interdicted_edges, interdicted_quantities):
                if (e, k) in gamma:
                    master.addConstr(gamma[e, k] == 1)

        # Mutually exclusive attempts per edge
        master.addConstrs((grb.quicksum(gamma[e, k] for k in k_vals) <= 1 
                          for e in self.interdictable_edges), name="one_k_per_edge")

        # Budget
        master.addConstr(grb.quicksum(self.edges_episode[e].interdiction_cost * k * gamma[e, k] 
                                      for e in self.interdictable_edges for k in k_vals) <= self.state['budget'][0], name="budget")
        
        master.setObjective(theta, grb.GRB.MINIMIZE)
        
        # 3. Subproblem Prep
        # Use centralized model via solve_max_flow
        
        # Benders Loop
        LB = -float('inf')
        UB = float('inf')
        epsilon = 1e-4
        max_iter = 100
        
        # Initialize x_hat
        x_hat = {}

        for iteration in range(max_iter):
            if time_limit is not None and time.time() - start_time > time_limit:
                break
                
            master.optimize()
            if master.status != grb.GRB.OPTIMAL:
                break
                
            x_hat = { (e,k): gamma[e,k].X for e,k in gamma_indices }
            current_theta = theta.X
            LB = current_theta 
            
            total_flow = 0.0
            cut_term_coefs = defaultdict(float) 
            
            for s in range(n_scenarios):
                # Update Capacities
                current_caps = {}
                for idx, e in enumerate(self.both_edges):
                    blocked = False
                    if e in interdictable_edge_map:
                         e_idx = interdictable_edge_map[e]
                         # Check blockade
                         for k in k_vals:
                             if x_hat.get((e, k), 0) > 0.5 and scenario_outcomes[s, e_idx, k-1] == 1:
                                 blocked = True
                                 break
                    
                    cap = 0 if blocked else self.edges_episode[e].capacity
                    current_caps[e] = cap
                    current_caps[(e[1], e[0])] = cap
                
                sub_obj, flow_dict = self.solve_max_flow(capacity_dict=current_caps)
                
                total_flow += sub_obj
                
                # Calculate Coefs
                for e in self.interdictable_edges:
                    # Get flow on edge
                    f_total = flow_dict.get(e, 0) + flow_dict.get((e[1], e[0]), 0)
                    e_idx = interdictable_edge_map[e]
                    
                    for k in k_vals:
                        # If outcome is success, we would block flow
                        if scenario_outcomes[s, e_idx, k-1] == 1:
                            cut_term_coefs[e, k] -= f_total

            avg_flow = total_flow / n_scenarios
            UB = avg_flow
            
            if avg_flow <= current_theta + epsilon:
                 break

            # Add Cut
            grad_dot_xhat = sum(cut_term_coefs[e, k] * x_hat[e, k] for e, k in gamma_indices)
            intercept = total_flow - grad_dot_xhat
            
            rhs = intercept / n_scenarios
            lhs_terms = grb.quicksum((cut_term_coefs[key]/n_scenarios) * gamma[key] for key in gamma_indices)
            
            master.addConstr(theta >= rhs + lhs_terms)
            master.update()

        # Extract Solution using x_hat
        interdicted = []
        quantities = []
        for e in self.interdictable_edges:
            for k in k_vals:
                if x_hat.get((e, k), 0) > 0.5:
                    interdicted.append(e)
                    quantities.append(k)
                    break 
                    
        return UB, interdicted, quantities

    def solve_stochastic_max_flow(self, n_scenarios = 50, seed = 173, interdicted_edges = [], method='monolithic', threads=None, time_limit=None):
        if method == 'decomposition':
            return self._solve_stochastic_decomposition(n_scenarios, seed, interdicted_edges, threads=threads, time_limit=time_limit)

        # Optimally Solve for Stochastic Solution using Model 1U and SAA
        if not hasattr(self, 'optimal_stochastic_model'):
            # Initializing the model
            self.optimal_stochastic_model = grb.Model("Stochastic Model", env=self.GUROBI_ENV)

        if threads is not None:
            self.optimal_stochastic_model.setParam("Threads", threads)
        if time_limit is not None:
            self.optimal_stochastic_model.setParam("TimeLimit", time_limit)

        if not hasattr(self, 'stochastic_gamma'):
            # Creating decision variables
            self.stochastic_gamma = self.optimal_stochastic_model.addVars(self.both_edges, vtype=grb.GRB.BINARY, name="gamma")

            # Create Variable Lower and Upper Bounds
            self.optimal_stochastic_model.setAttr("LB", [self.stochastic_gamma[e] for e in interdicted_edges],1)
            self.optimal_stochastic_model.setAttr("UB", [self.stochastic_gamma[e] for e in self.noninterdictable_edges],0)
            
             # Gamma constraint
            self.stochastic_gamma_constr = self.optimal_stochastic_model.addConstrs((self.stochastic_gamma[e] <= 1 for e in self.both_edges), name="gamma_constr")
            
             # Budget constraint
            self.stochastic_budget_constr = self.optimal_stochastic_model.addConstr(grb.quicksum(self.edges_episode[e].interdiction_cost * self.stochastic_gamma[e] for e in self.both_edges) <= self.state['budget'][0], name="budget")

            self.stochastic_old_state = self.state
            self.stochastic_old_interdicted_edges = interdicted_edges

        if self.stochastic_old_interdicted_edges != interdicted_edges:
            # Update Variable Lower and Upper Bounds
            self.optimal_stochastic_model.setAttr("LB", [self.stochastic_gamma[e] for e in self.both_edges],0)
            self.optimal_stochastic_model.setAttr("LB", [self.stochastic_gamma[e] for e in interdicted_edges],1)
            self.stochastic_old_interdicted_edges=interdicted_edges
        
        if self.num_stochastic_scenarios != n_scenarios:
            # Generate scenarios
            self.num_stochastic_scenarios = n_scenarios
            self.scenarios = range(n_scenarios)

            if hasattr(self, 'stochastic_alpha'):
                self.optimal_stochastic_model.remove(self.stochastic_alpha)
                self.optimal_stochastic_model.remove(self.stochastic_beta) #Remove from model to prevent errors
                self.optimal_stochastic_model.update()  # Force model synchronization
                del self.stochastic_alpha, self.stochastic_beta 
                
            self.stochastic_alpha = self.optimal_stochastic_model.addVars([(i, s) for s in self.scenarios for i in self.nodes], 
                                                  vtype=grb.GRB.BINARY, name="alpha")
            self.stochastic_beta = self.optimal_stochastic_model.addVars([(e, s) for s in self.scenarios for e in self.edges_reset],
                                                                          vtype=grb.GRB.BINARY, name="beta")
            self.optimal_stochastic_model.update()

            if hasattr(self, 'stochastic_source_sink_constr'):
                self.optimal_stochastic_model.remove(self.stochastic_source_sink_constr)
                del self.stochastic_source_sink_constr

            self.stochastic_source_sink_constr = self.optimal_stochastic_model.addConstrs((self.stochastic_alpha[self.super_sink_nodes[0],s] - self.stochastic_alpha[self.super_source_nodes[0], s] >= 1 for s in self.scenarios), name="source_sink")

            # Objective Function
            self.optimal_stochastic_model.setObjective((1/n_scenarios)*grb.quicksum(self.edges_episode[e].capacity * self.stochastic_beta[e, s]
                for s in self.scenarios
                for e in self.edges_reset), grb.GRB.MINIMIZE)
        
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
            (self.stochastic_alpha[e[1],s] - self.stochastic_alpha[e[0], s] + self.stochastic_beta[e, s] + (self.stochastic_gamma[e] * scenario_outcomes[s, edge_id]) >= 0 for s in self.scenarios for edge_id, e in enumerate(self.both_edges)), name='aabg_reverse')

        # Solving
        self.optimal_stochastic_model.optimize()

        interdicted_edges = [
            e for e in self.both_edges
            if self.stochastic_gamma[e].X > 0.5 
        ]

        return(self.optimal_stochastic_model.objVal, interdicted_edges)


    def _generate_exact_scenarios_and_probs(self, max_scenarios=200):
        """
        Helper method to generate all possible scenarios and their probabilities 
        for edges with 0 < probability < 1.
        Returns:
            scenario_outcomes: List of binary vectors (one per scenario) for all edges
            scenario_probs: List of probabilities for each scenario
        """
        # 1. Identify relevant edges (prob > 0 and prob < 1)
        probs = self.state["edge_interdiction_probability"][:self.num_both_edges]
        stochastic_indices = [i for i, p in enumerate(probs) if 0 < p < 1]
        
        # 2. Generate Scenarios
        # If too many scenarios, we use a heap to keep only the top N most likely
        n_exact = 2**len(stochastic_indices)
        
        if n_exact > 20000:
            logging.warning(f"Too many scenarios ({n_exact}). Limiting to top {max_scenarios} most likely.")
            import heapq
            
            # Helper to calculate prob of a partial/full outcome
            # We want to find top scenarios without iterating all 2^22
            # But implementing an efficient "next best" search is complex.
            # For 2^22 (4M), we can iterate comfortably in C++, but in Python it might take ~10-20s.
            # Let's try to iterate but use a min-heap to track top N.
            
            # Optimization: If p > 0.5 for most edges, the mass is concentrated on "success".
            # If we iterate all, it takes time.
            # Alternative: Just sample? No, user asked for "most likely".
            
            # Let's use a priority queue search to find top N scenarios
            # Start with the most likely scenario (for each edge, pick outcome with max p)
            # Then explore neighbors by flipping state of one edge
            
            scenario_outcomes_list = []
            scenario_probs = []
            
            # 1. Construct Most Likely Scenario (Base)
            best_outcome_bits = []
            current_log_prob = 0.0 # Working in log space for numerical stability is better, but simple probs ok
            
            # Precompute log probs for stability
            log_p_success = []
            log_p_fail = []
            
            for idx in stochastic_indices:
                p = probs[idx]
                # Avoid log(0)
                lp_s = math.log(p) if p > 1e-9 else -1000.0
                lp_f = math.log(1-p) if (1-p) > 1e-9 else -1000.0
                log_p_success.append(lp_s)
                log_p_fail.append(lp_f)
            
            # Determine best state for each bit
            initial_state = []
            initial_log_prob = 0.0
            
            # We need to compute delta for flipping each bit from best to second best
            deltas = [] # (cost_to_flip, bit_index)
            
            for i in range(len(stochastic_indices)):
                ls = log_p_success[i]
                lf = log_p_fail[i]
                
                if ls >= lf:
                    initial_state.append(1)
                    initial_log_prob += ls
                    diff = ls - lf # Positive cost to flip to fail
                    deltas.append((diff, i, 0)) # 0 means "flip to 0"
                else:
                    initial_state.append(0)
                    initial_log_prob += lf
                    diff = lf - ls
                    deltas.append((diff, i, 1)) # 1 means "flip to 1"
            
            deltas.sort(key=lambda x: x[0]) # Sort by cost to flip (ascending)
            
            # Base outcome initialized with deterministic edges
            # Edges with p=1 are 1, others 0 (stochastic ones will be filled)
            base_outcome = [1 if p >= 0.999 else 0 for p in probs]
            
            # Priority Queue for searching states: (-log_prob, state_tuple)
            # Actually we can just search in the "delta space".
            # State is defined by set of indices flipped from optimal.
            # Queue stores: (current_penalty, last_index_in_deltas_processed, active_flips_indices)
            
            # 0. Best scenario
            pq = [(0.0, -1, ())] # penalty, max_idx_used, indices_flipped
            
            count = 0
            
            # Top N Search
            while count < max_scenarios and pq:
                penalty, max_idx, flipped_tuple = heapq.heappop(pq)
                
                # Reconstruct this scenario
                current_outcome = base_outcome.copy() # Contains deterministic 1s
                prob_val = math.exp(initial_log_prob - penalty)
                
                # Apply optimal baselines for stochastic
                local_stochastic_outcomes = list(initial_state)
                
                # Apply flips
                for flip_idx_ptr in flipped_tuple:
                    # deltas[flip_idx_ptr] is (diff, original_index, target_val)
                    diff, orig_idx, target_val = deltas[flip_idx_ptr]
                    local_stochastic_outcomes[orig_idx] = target_val
                
                # Fill current_outcome
                for i, idx in enumerate(stochastic_indices):
                    current_outcome[idx] = local_stochastic_outcomes[i]
                
                scenario_outcomes_list.append(current_outcome)
                scenario_probs.append(prob_val)
                count += 1
                
                # Generate successors
                # 1. Extend current flip set with next available delta
                next_idx = max_idx + 1
                if next_idx < len(deltas):
                    # Child 1: Add next delta to current set
                    new_penalty = penalty + deltas[next_idx][0]
                    new_flips = flipped_tuple + (next_idx,)
                    heapq.heappush(pq, (new_penalty, next_idx, new_flips))
                    
                    # Child 2: Replace last delta with next delta (if not root)
                    if flipped_tuple:
                         # Remove last added, add next
                         prev_penalty_contrib = deltas[max_idx][0]
                         new_node_penalty = penalty - prev_penalty_contrib + deltas[next_idx][0]
                         # Pop last from tuple
                         new_node_flips = flipped_tuple[:-1] + (next_idx,)
                         heapq.heappush(pq, (new_node_penalty, next_idx, new_node_flips))
            
            # Re-normalize probabilities
            total_p = sum(scenario_probs)
            print(f"Top {len(scenario_probs)} scenarios cover {total_p:.6f} probability mass. Unaccounted: {1.0 - total_p:.6f}")
            if total_p > 0:
                scenario_probs = [p / total_p for p in scenario_probs]
                
            return scenario_outcomes_list, scenario_probs

        else:
            # Original Exact Logic
            # Generate combinations of 0/1 for the stochastic edges
            outcomes_combinations = list(itertools.product([0, 1], repeat=len(stochastic_indices)))
            
            # Base outcome initialized with deterministic edges
            base_outcome = [1 if p >= 0.999 else 0 for p in probs]
            
            scenario_outcomes_list = []
            scenario_probs = []
            
            for outcome in outcomes_combinations:
                current_outcome = base_outcome.copy()
                scenario_prob = 1.0
                
                for i, idx in enumerate(stochastic_indices):
                    is_success = outcome[i]
                    # ... rest of original loop ...
                    current_outcome[idx] = is_success
                    
                    p = probs[idx]
                    if is_success:
                        scenario_prob *= p
                    else:
                        scenario_prob *= (1 - p)
                
                scenario_outcomes_list.append(current_outcome)
                scenario_probs.append(scenario_prob)
                
            # Sort and clip if we are in the 20-20000 range but user wants top 200 explicitly?
            # The condition above handles > 20000. 
            # If < 20000 but > 200, we currently keep all.
            # User asked "limit it to 200". We should apply it generally?
            # Let's apply sorting and clipping if n > max_scenarios
            
            if len(scenario_probs) > max_scenarios:
                # Zip, sort, unzip
                zipped = sorted(zip(scenario_probs, scenario_outcomes_list), key=lambda x: -x[0])
                scenario_probs = [p for p, o in zipped[:max_scenarios]]
                scenario_outcomes_list = [o for p, o in zipped[:max_scenarios]]
                
                # Normalize
                total_p = sum(scenario_probs)
                print(f"Top {len(scenario_probs)} scenarios cover {total_p:.6f} probability mass. Unaccounted: {1.0 - total_p:.6f}")
                if total_p > 0:
                    scenario_probs = [p / total_p for p in scenario_probs]

            return scenario_outcomes_list, scenario_probs

    def solve_exact_monolithic(self, max_scenarios=200):
        """
        Solves the stochastic updated interdiction problem exactly using a monolithic MIP formulation.
        Enumerates all outcome scenarios weighted by their probability.
        """
        # 1. Generate Scenarios
        scenario_outcomes_list, scenario_probs = self._generate_exact_scenarios_and_probs(max_scenarios=max_scenarios)
        n_scenarios = len(scenario_probs)
        scenarios = range(n_scenarios)

        # 2. Build Model
        # Create a local environment to ensure output is captured in Jupyter
        monolithic_env = grb.Env(empty=True)
        monolithic_env.setParam("OutputFlag", 1)
        monolithic_env.setParam("LogToConsole", 0) # Disable C-level logging to avoid duplication
        monolithic_env.start()
        
        model = grb.Model("Exact_Monolithic_Stochastic", env=monolithic_env)
        model.setParam("Threads", 0)     # Use all available threads
        
        # Decision Variables
        gamma = model.addVars(self.both_edges, vtype=grb.GRB.BINARY, name="gamma")
        
        # Budget Constraint
        model.addConstr(
            grb.quicksum(self.edges_episode[e].interdiction_cost * gamma[e] for e in self.both_edges) 
            <= self.state['budget'][0], name="budget"
        )
        
        # Non-interdictable constraints
        model.setAttr("UB", [gamma[e] for e in self.noninterdictable_edges], 0)
        
        # Scenario-specific Dual Variables (Alpha, Beta)
        alpha = model.addVars([(i, s) for s in scenarios for i in self.nodes], 
                                     vtype=grb.GRB.BINARY, name="alpha")
        beta = model.addVars([(e, s) for s in scenarios for e in self.edges_reset],
                                    vtype=grb.GRB.BINARY, name="beta")
        
        # Source-Sink cut constraint per scenario
        model.addConstrs(
            (alpha[self.super_sink_nodes[0], s] - alpha[self.super_source_nodes[0], s] >= 1 
             for s in scenarios), name="source_sink"
        )
        
        # AABG Constraints (Dual Max Flow) weighted by outcome
        for s in scenarios:
            outcome_vec = scenario_outcomes_list[s]
            for idx, e in enumerate(self.both_edges):
                outcome_val = outcome_vec[idx]
                # Forward
                model.addConstr(
                    alpha[e[0], s] - alpha[e[1], s] + beta[e, s] + (gamma[e] * outcome_val) >= 0
                )
                # Reverse (assuming undirected edge logic or separate reverse edges handling)
                # The existing code typically adds reverse constraints for undirected edges if they are modeled as pairs
                model.addConstr(
                    alpha[e[1], s] - alpha[e[0], s] + beta[e, s] + (gamma[e] * outcome_val) >= 0
                )

        # Objective: Min Expected Max Flow
        obj_expr = 0
        for s in scenarios:
             s_prob = scenario_probs[s]
             s_cut_val = grb.quicksum(self.edges_episode[e].capacity * beta[e, s] for e in self.edges_reset)
             obj_expr += s_prob * s_cut_val
             
        model.setObjective(obj_expr, grb.GRB.MINIMIZE)
        
        # Optimize with callback for Jupyter output
        import sys
        def jupyter_callback(model, where):
            if where == grb.GRB.Callback.MESSAGE:
                # Capture message
                msg = model.cbGet(grb.GRB.Callback.MSG_STRING)
                if msg:
                    # Write to both stdout and stderr to ensure visibility
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                
        print("Starting Monolithic Optimization (Exact)...", file=sys.stdout)
        sys.stdout.flush()
        
        # Pass callback to optimize
        model.optimize(jupyter_callback)
        
        print(f"Optimization Finished. Objective: {model.objVal}", file=sys.stdout)
        sys.stdout.flush()
        
        interdicted = [e for e in self.both_edges if gamma[e].X > 0.5]
        
        # Use centralized evaluation method to ensure consistency
        final_obj_val = self._evaluate_solution_stochastic(interdicted)
        
        return final_obj_val, interdicted

    def solve_exact_decomposition(self, max_scenarios=200):
        """
        Solves the stochastic updated interdiction problem exactly using Benders Decomposition.
        Enumerates all outcome scenarios weighted by their probability.
        """
        # 1. Generate Scenarios
        scenario_outcomes_list, scenario_probs = self._generate_exact_scenarios_and_probs(max_scenarios=max_scenarios)
        n_scenarios = len(scenario_probs)
        
        # 2. Master Problem
        # Create a local environment to ensure output is captured in Jupyter
        benders_env = grb.Env(empty=True)
        benders_env.setParam("OutputFlag", 1)
        benders_env.setParam("LogToConsole", 0) # Disable C-level logging to avoid duplication
        benders_env.start()

        master = grb.Model("Exact_Benders_Master", env=benders_env)
        master.setParam("Threads", 0)     # Use all available threads
        
        gamma = master.addVars(self.both_edges, vtype=grb.GRB.BINARY, name="gamma")
        theta = master.addVar(lb=0, name="theta")
        
        # Budget
        master.addConstr(
            grb.quicksum(self.edges_episode[e].interdiction_cost * gamma[e] for e in self.both_edges) 
            <= self.state['budget'][0], name="budget"
        )
        # Non-interdictable constraints
        master.setAttr("UB", [gamma[e] for e in self.noninterdictable_edges], 0)
        
        master.setObjective(theta, grb.GRB.MINIMIZE)
        
        # 3. Benders Loop
        epsilon = 1e-4
        max_iter = 100
        UB = float('inf')
        
        # Initialize x_hat
        x_hat = {} 
        
        # Callback for Jupyter output
        import sys
        def jupyter_callback(model, where):
            if where == grb.GRB.Callback.MESSAGE:
                msg = model.cbGet(grb.GRB.Callback.MSG_STRING)
                if msg:
                     sys.stdout.write(msg)
                     sys.stdout.flush()
        
        for iteration in range(max_iter):
            print(f"\n--- Benders Iteration {iteration+1} ---", file=sys.stdout)
            sys.stdout.flush()
            master.optimize(jupyter_callback)
            if master.status != grb.GRB.OPTIMAL:
                break
                
            x_hat = {e: gamma[e].X for e in self.both_edges}
            current_theta = theta.X
            
            # Solve Subproblems (Weighted by Probability)
            total_expected_flow = 0.0
            cut_term_coefs = defaultdict(float) # Coef for gamma[e]
            
            for s in range(n_scenarios):
                s_prob = scenario_probs[s]
                s_outcome = scenario_outcomes_list[s]
                
                # Update Capacities based on x_hat and outcome
                current_caps = {}
                for idx, e in enumerate(self.both_edges):
                    # Interdiction is successful if attempted (x_hat > 0.5) AND outcome is 1
                    is_interdicted = (x_hat[e] > 0.5)
                    outcome_success = (s_outcome[idx] == 1)
                    
                    is_blocked = is_interdicted and outcome_success
                    cap = 0 if is_blocked else self.edges_episode[e].capacity
                    current_caps[e] = cap
                    rev_e = (e[1], e[0])
                    current_caps[rev_e] = cap
                
                # Solve max flow
                sub_obj, flow_dict = self.solve_max_flow(capacity_dict=current_caps)
                
                total_expected_flow += s_prob * sub_obj
                
                # Calculate cut coefficients
                for idx, e in enumerate(self.both_edges):
                     if s_outcome[idx] == 1:
                         # Flow that would be blocked
                         f_fwd = flow_dict.get(e, 0)
                         rev_e = (e[1], e[0])
                         f_rev = flow_dict.get(rev_e, 0)
                         f_total = f_fwd + f_rev
                         
                         # Coefficient: - Prob * Flow * Outcome (outcome is 1 here)
                         cut_term_coefs[e] -= s_prob * f_total

            UB = total_expected_flow
            
            if UB <= current_theta + epsilon:
                 break
                 
            # Add Weighted Cut
            # Theta >= Sum( P_s * MaxFlow_s(x_hat) ) + Sum_e ( Sum_s (P_s * dMaxFlow/dx_e) * (gamma_e - x_hat_e) )
            # Simplified: Theta >= Intercept + Sum(Coef_e * gamma_e)
            
            grad_dot_xhat = sum(cut_term_coefs[e] * x_hat[e] for e in self.both_edges)
            intercept = total_expected_flow - grad_dot_xhat
            
            lhs_terms = grb.quicksum(cut_term_coefs[e] * gamma[e] for e in self.both_edges)
            master.addConstr(theta >= intercept + lhs_terms)
            master.update()
            
        interdicted = [e for e in self.both_edges if x_hat.get(e, 0) > 0.5]
        
        # Use centralized evaluation method to ensure consistency
        # Note: UB here is the expected value across the scenarios used in decomposition
        # Evaluating with _evaluate_solution_stochastic will use the default sampling/exact method of the environment
        final_obj_val = self._evaluate_solution_stochastic(interdicted)
        
        return final_obj_val, interdicted

    def _compute_baycik_static_features(self):
        """Compute static topological features for Baycik's methodology."""
        # 1. Degrees
        in_degrees = {n: len(self.edge_groups[n]['in']) for n in self.nodes}
        out_degrees = {n: len(self.edge_groups[n]['out']) for n in self.nodes}
        
        # 2. Distance from Source (BFS)
        dist_from_source = {1: 0} # 1 is super source
        max_dist_src = 1.0
        queue = deque([1])
        visited_src = {1}
        
        while queue:
            u = queue.popleft()
            d = dist_from_source[u]
            max_dist_src = max(max_dist_src, d)
            
            # Outbound neighbors
            for _, v in self.edge_groups[u]['out']:
                if v not in visited_src:
                    visited_src.add(v)
                    dist_from_source[v] = d + 1.0
                    queue.append(v)
                    
        # 3. Distance to Sink (Reverse BFS)
        sink = self.super_sink_nodes[0]
        dist_to_sink = {sink: 0}
        max_dist_sink = 1.0
        queue = deque([sink])
        visited_sink = {sink}
        
        while queue:
            v = queue.popleft()
            d = dist_to_sink[v]
            max_dist_sink = max(max_dist_sink, d)
            
            # Inbound neighbors (reverse graph)
            for u, _ in self.edge_groups[v]['in']:
                if u not in visited_sink:
                    visited_sink.add(u)
                    dist_to_sink[u] = d + 1.0
                    queue.append(u)
                    
        return {
            'in_degree': in_degrees,
            'out_degree': out_degrees,
            'dist_src': dist_from_source,
            'dist_sink': dist_to_sink,
            'max_dist_src': max_dist_src,
            'max_dist_sink': max_dist_sink
        }

    def train_baycik_model(self, pickle_path, start_index=0, end_index=None):
        """Train Random Forest model using Baycik's methodology from solution file."""
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            
        states = data['states']
        optimal_solutions = data['all_optimal_interdiction_edges']
        
        X = []
        y = []
        
        # Compute static features once
        static_feats = self._compute_baycik_static_features()
        
        # Save current state
        original_state = copy.deepcopy(self.state)
        
        try:
            # We iterate through the saved episodes
            # Note: We limit to valid episodes (where state is not None)
            limit = end_index if end_index is not None else len(states)
            for i in tqdm(range(start_index, limit), desc="Training RF"):
                state = states[i]
                if state is None: continue
                
                # Handle Gymnasium reset return (obs, info)
                if isinstance(state, tuple) and len(state) == 2:
                    if isinstance(state[0], dict) and 'edge_capacity' in state[0]:
                        state = state[0]

                # Load state into environment
                self.state = state
                self._cache_flow_array() # Update cache based on state
                
                # Initialize reference values for the current attacker strategy (needed for training consistency)
                if self.attacker_strategy == 'canalize':
                    _, flows = self.solve_max_flow(routing_assumption = 'canalize')
                    self.reference_start_flow = self._calculate_target_path_flow(flows, 'canalize_objective')
                elif self.attacker_strategy == 'divert':
                    _, flows = self.solve_max_flow(routing_assumption = 'divert')
                    from_flow = self._calculate_target_path_flow(flows, 'divert_from_objective')
                    to_flow = self._calculate_target_path_flow(flows, 'divert_to_objective')
                    self.reference_start_flows = (from_flow, to_flow)

                # Calculate Initial Uninterdicted Flow
                # Ensure no interdictions are considered for the feature extraction
                # We temporarily clear interdictions in state dict
                temp_interdicted = self.state['edge_interdicted'].copy()
                self.state['edge_interdicted'] = np.zeros_like(temp_interdicted)
                
                # Solve max flow to get features
                # _, flow_dict = self.solve_max_flow()
                
                # Restore interdictions (though usually 0 at start of episode)
                self.state['edge_interdicted'] = temp_interdicted
                
                budget = state['budget'][0]
                target_set = set(optimal_solutions[i])
                
                # Max capacity for normalization
                current_caps = state['edge_capacity'][:self.num_both_edges]
                max_net_cap = np.max(current_caps) if len(current_caps) > 0 else 1.0
                
                # Iterative Data Expansion:
                # Iterate until all target edges are added to interdiction set
                # Recompute features after each addition
                
                # Sort target set by some heuristic if needed, or arbitrary.
                # Here we just iterate through them arbitrarily (set order).
                # Actually, to be robust, we treat all remaining targets as positives (1)
                # and pick one to "execute" to move to next state.
                
                targets_remaining = list(target_set)
                random.shuffle(targets_remaining) # Shuffle to randomize trajectory order
                
                current_interdicted_indices = set() # Track what we've added in this expansion loop
                
                # Outer loop: State trajectory (0 interdicted -> 1 -> ... -> N-1)
                # We stop when 1 target remains? No, we can train on the last step too (picking the last one).
                # So we loop len(targets) times.
                
                for step_idx in range(len(targets_remaining) ):
                    # 1. Update Strategy Flows and calculate features
                    obj_iso_can = 0.0
                    obj_div_from = 0.0
                    obj_div_to = 0.0

                    try:
                        if self.attacker_strategy == "zero_sum":
                            _, flow_dict = self._compute_objective_and_flows()
                        elif self.attacker_strategy == "canalize":
                            obj_iso_can, flow_dict = self._calculate_canalize_objective_and_flows()
                        elif self.attacker_strategy == "isolate":
                            obj_iso_can, flow_dict = self._calculate_isolate_objective_and_flows()
                        elif self.attacker_strategy == "divert":
                            _, flow_dict = self._calculate_divert_objective_and_flows()
                            # Use dict flows for components
                            _, flows_dict = self.solve_max_flow(routing_assumption = 'divert')
                            from_flow = self._calculate_target_path_flow(flows_dict, 'divert_from_objective')
                            to_flow = self._calculate_target_path_flow(flows_dict, 'divert_to_objective')
                            obj_div_from = (getattr(self, 'reference_start_flows', [0,0])[0] - from_flow)
                            obj_div_to = (to_flow - getattr(self, 'reference_start_flows', [0,0])[1])
                        else:
                            _, flow_dict = self.solve_max_flow()
                    except Exception:
                        _, flow_dict = self.solve_max_flow()
                    
                    current_budget = self.state['budget'][0]
                    
                    # 2. Collect samples for all edges
                    for idx in range(self.num_both_edges):
                        # Skip edges already interdicted in this trajectory
                        # (The environment handles the logic, but for feature extraction loop)
                        if self.state['edge_interdicted'][idx] == 1:
                            continue
                            
                        edge = self.both_edges[idx]
                        u, v = edge
                        
                        cap = self.state['edge_capacity'][idx]
                        cost = self.state['edge_costs'][idx]
                        
                        if isinstance(flow_dict, np.ndarray):
                            f_val = flow_dict[idx]
                        else:
                            f_val = flow_dict.get(edge, 0) + flow_dict.get((edge[1],edge[0]), 0)

                        
                        # Extra Features
                        tail_in = static_feats['in_degree'].get(u, 0)
                        tail_out = static_feats['out_degree'].get(u, 0)
                        head_in = static_feats['in_degree'].get(v, 0)
                        head_out = static_feats['out_degree'].get(v, 0)
                        
                        d_src = static_feats['dist_src'].get(u, static_feats['max_dist_src'])
                        d_sink = static_feats['dist_sink'].get(v, static_feats['max_dist_sink'])
                        
                        norm_d_src = d_src / static_feats['max_dist_src']
                        norm_d_sink = d_sink / static_feats['max_dist_sink']
                        norm_cap = cap / max_net_cap
                        
                        prob_success = self.state['edge_interdiction_probability'][idx]
                            
                        # Features: Cost, Flow, Budget, Interdiction Prob + 7 Static + 3 Strategy.
                        features = [cost, f_val, current_budget, prob_success,
                            tail_in, tail_out, head_in, head_out,
                            norm_d_src, norm_d_sink, norm_cap,
                            obj_iso_can, obj_div_from, obj_div_to]

                        # Label is 1 if edge is in the remaining target set
                        label = 1 if edge in target_set and edge not in current_interdicted_indices else 0
                        
                        X.append(features)
                        y.append(label)
                    
                    # 3. Transition: Pick next edge from targets to interdict
                    if step_idx < len(targets_remaining):
                        next_edge = targets_remaining[step_idx]
                        next_idx = self.edge_to_index[next_edge]
                        
                        # Execute step (updates budget, interdicted status, probabilistically?)
                        # Since this is training on "optimal" ground truth which assumes successful interdiction usually,
                        # or at least the attempt.
                        # We use env.step() to handle budget and state updates correctly.
                        # Note: step() might fail interdiction if prob < 1. 
                        # But optimal_solutions usually refers to the set of edges *attempted* that yielded result?
                        # Or the edges that *should* be cut.
                        # We force success for feature calculation purposes if we want to simulate the "ideal" trajectory,
                        # OR we use step() and handle the stochasticity.
                        # Given 'optimal_solutions' comes from a static solution (likely), we should force it.
                        
                        # Direct state update to force "interdicted" behavior for feature flow calc
                        self.state['edge_interdicted'][next_idx] = 1
                        self.state['budget'][0] -= self.state['edge_costs'][next_idx]
                        current_interdicted_indices.add(next_edge)
                        
                        # If budget runs out, stop trajectory
                        if self.state['budget'][0] < 0:
                            break
                    
        finally:
            self.state = original_state
            self._cache_flow_array()
            
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        return clf

    def solve_baycik_interdiction(self, model):
        """Solve using Baycik's Random Forest Heuristic with dynamic re-evaluation."""
        # Static Features
        static_feats = self._compute_baycik_static_features()
        
        # Max capacity for normalization
        current_caps = self.state['edge_capacity'][:self.num_both_edges]
        max_net_cap = np.max(current_caps) if len(current_caps) > 0 else 1.0
        
        selected_edges = []
        
        # Save state before greedy rollout
        saved_budget = self.state['budget'][0]
        saved_interdicted = self.state['edge_interdicted'].copy()
        
        try:
            # Greedy Selection (Masked Rollout)
            done = False
            selected_indices = set()
            
            while not done:
                # 1. Update Strategy Flows and calculate extra features
                obj_iso_can = 0.0
                obj_div_from = 0.0
                obj_div_to = 0.0

                try:
                    if self.attacker_strategy == "zero_sum":
                        _, self.reference_flows = self._compute_objective_and_flows()
                    elif self.attacker_strategy == "canalize":
                        obj_iso_can, self.reference_flows = self._calculate_canalize_objective_and_flows()
                    elif self.attacker_strategy == "isolate":
                        obj_iso_can, self.reference_flows = self._calculate_isolate_objective_and_flows()
                    elif self.attacker_strategy == "divert":
                        _, self.reference_flows = self._calculate_divert_objective_and_flows()
                        # Use dict flows for components
                        _, flows_dict = self.solve_max_flow(routing_assumption = 'divert')
                        from_flow = self._calculate_target_path_flow(flows_dict, 'divert_from_objective')
                        to_flow = self._calculate_target_path_flow(flows_dict, 'divert_to_objective')
                        obj_div_from = (getattr(self, 'reference_start_flows', [0,0])[0] - from_flow)
                        obj_div_to = (to_flow - getattr(self, 'reference_start_flows', [0,0])[1])
                except Exception:
                    # Fallback check
                    pass

                self._cache_flow_array()
                mask = self.mask_fn()
                
                # 2. Update Feature Flows (Standard Zero Sum for Features consistency with Train)
                # If strategy is zero_sum, we technically computed this above.
                flow_dict = self.reference_flows
                                
                current_budget = self.state['budget'][0]
                
                best_candidate_idx = -1
                best_prob = -1.0
                
                # 3. Score candidates
                for idx in range(self.num_both_edges):
                    if mask[idx] == 0: continue
                    if idx in selected_indices: continue
                        
                    edge = self.both_edges[idx]
                    u, v = edge
                    
                    cap = self.state['edge_capacity'][idx]
                    cost = self.state['edge_costs'][idx]
                    
                    # Consistent with train_baycik_model
                    if isinstance(flow_dict, np.ndarray):
                        f_val = flow_dict[idx]
                    else:
                        f_val = flow_dict.get(edge, 0) + flow_dict.get((edge[1],edge[0]), 0)
                    
                    tail_in = static_feats['in_degree'].get(u, 0)
                    tail_out = static_feats['out_degree'].get(u, 0)
                    head_in = static_feats['in_degree'].get(v, 0)
                    head_out = static_feats['out_degree'].get(v, 0)
                    
                    d_src = static_feats['dist_src'].get(u, static_feats['max_dist_src'])
                    d_sink = static_feats['dist_sink'].get(v, static_feats['max_dist_sink'])
                    
                    norm_d_src = d_src / static_feats['max_dist_src']
                    norm_d_sink = d_sink / static_feats['max_dist_sink']
                    norm_cap = cap / max_net_cap
                    
                    prob_success = self.state['edge_interdiction_probability'][idx]

                    # Consistent Order with train_baycik_model: Cost, Flow, Budget, ..., Strategy Obj
                    features = [cost, f_val, current_budget, prob_success,
                                tail_in, tail_out, head_in, head_out,
                                norm_d_src, norm_d_sink, norm_cap,
                                obj_iso_can, obj_div_from, obj_div_to]
                    
                    # Predict
                    prob = model.predict_proba([features])[0][1]
                    
                    if prob > best_prob:
                        best_prob = prob
                        best_candidate_idx = idx

                if best_candidate_idx != -1:
                    selected_edges.append(self.both_edges[best_candidate_idx])
                    selected_indices.add(best_candidate_idx)
                    _, _, done, _, _ = self.step(best_candidate_idx)
                else:
                    break
                    
        finally:
            # Restore state
            self.state['budget'][0] = saved_budget
            self.state['edge_interdicted'] = saved_interdicted
            self._cache_flow_array()

        # Evaluate
        objective_value = self._evaluate_solution_stochastic(selected_edges)
            
        return objective_value, selected_edges

    def solve_stochastic_max_flow_IM(self, n_scenarios = 50, seed = 173, interdicted_edges = [], interdicted_quantities =[], method='monolithic', threads=None, time_limit=None):
        if method == 'decomposition':
            return self._solve_stochastic_decomposition_IM(n_scenarios, seed, interdicted_edges, interdicted_quantities, threads=threads, time_limit=time_limit)

        # Optimally Solve for Stochastic Solution using Model 1D and SAA
        if not hasattr(self, 'optimal_stochastic_model_IM'):
            # Initializing the model
            self.optimal_stochastic_model_IM = grb.Model("Stochastic Model_IM", env=self.GUROBI_ENV)

        if threads is not None:
            self.optimal_stochastic_model_IM.setParam("Threads", threads)
        if time_limit is not None:
            self.optimal_stochastic_model_IM.setParam("TimeLimit", time_limit)
        
        if not hasattr(self, 'stochastic_gamma_IM'):
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
            self.optimal_stochastic_model_IM.update()

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
        self.stochastic_aabg_reverse_constr_IM = self.optimal_stochastic_model_IM.addConstrs((self.stochastic_alpha_IM[e[1],s] - self.stochastic_alpha_IM[e[0], s]+self.stochastic_beta_IM[e, s]+ (grb.quicksum(self.stochastic_gamma_IM[e,k] * scenario_outcomes[s, interdictable_edge_map[e], k-1] for k in k_vals) if e in interdictable_edge_map else 0) >= 0 for s in self.scenarios_IM for e in self.edges_reset.keys()), name='aabg_reverse_IM')

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

    def _calculate_projections(self, valid_actions, flows, remaining_budget, projection_uses_flow):
        projected_values = np.zeros(len(valid_actions))

        # --- OPTIMIZATION 1: Cache static structural masks ---
        if not hasattr(self, '_static_proj_mask'):
            has_prob = self.state['edge_interdiction_probability'][:self.num_interdictable] > 0
            if self.attacker_strategy == 'canalize':
                strategy_mask = self.state['canalize_objective'][:self.num_interdictable] != 1
            elif self.attacker_strategy == 'divert':
                strategy_mask = self.state['divert_to_objective'][:self.num_interdictable] != 1
            else:
                strategy_mask = np.ones(self.num_interdictable, dtype=bool)
            self._static_proj_mask = has_prob & strategy_mask

        costs = self.state['edge_costs'][:self.num_interdictable]
        limit_ok = (self.state['edge_interdicted'][:self.num_interdictable] + 1) <= self.max_interdictions
        affordable = costs <= remaining_budget
        
        projection_mask = self._static_proj_mask & limit_ok & affordable
        
        # 2. Get Sorted Future Benefits
        projection_indices = np.where(projection_mask)[0]
        
        if len(projection_indices) > 0:
            # OPTIMIZATION: Use pre-computed base values if available
            if not projection_uses_flow and hasattr(self, 'heuristic_base_values'):
                benefits = self.heuristic_base_values[projection_indices]
            elif projection_uses_flow:
                probs_proj = self.state['edge_interdiction_probability'][projection_indices]
                
                if isinstance(flows, np.ndarray):
                   proj_flow_vals = flows[projection_indices]
                elif hasattr(self, 'cached_flow_array') and flows is getattr(self, 'reference_flows', None):
                   proj_flow_vals = self.cached_flow_array[projection_indices].sum(axis=1)
                else:
                   proj_flow_vals = np.array([
                       flows.get(self.both_edges[a], 0) + flows.get((self.both_edges[a][1], self.both_edges[a][0]), 0)
                       for a in projection_indices
                   ])
                benefits = proj_flow_vals * probs_proj
            else:
                caps_proj = self.state['edge_capacity'][projection_indices]
                probs_proj = self.state['edge_interdiction_probability'][projection_indices]
                benefits = caps_proj * probs_proj
            
            # Optimization: Only sort the projection elements we might actually use
            # --- OPTIMIZATION 3: Speedup Floor operation using int-division ---
            max_required_k = int((remaining_budget + 1e-9) // self.min_edge_cost) + 1
            max_available_k = min(max_required_k, len(benefits))
            
            if max_available_k > 0 and max_available_k < len(benefits):
                # Use argpartition to find the top max_available_k elements in O(N) time
                part_idx = np.argpartition(-benefits, max_available_k - 1)[:max_available_k]
                # Sort only the top partitioned elements
                top_sort_idx = np.argsort(-benefits[part_idx])
                sorted_idx = part_idx[top_sort_idx]
            else:
                # Fallback to full sort if k is near the array size
                sorted_idx = np.argsort(-benefits)
                
            sorted_benefits = benefits[sorted_idx]
            sorted_global_indices = projection_indices[sorted_idx]
            
            # Rank Lookup (Map global edge index -> rank in sorted list)
            # --- OPTIMIZATION 2: Direct mask evaluation & dropping redundant edge checks ---
            rank_lookup = np.full(int(self.num_both_edges), -1, dtype=int)
            rank_lookup[sorted_global_indices] = np.arange(len(sorted_global_indices))
            
            # Precompute cumulative sum (prefix sum) for fast range summation
            # cumsum[k] = sum of top k items
            cumsum_benefits = np.zeros(len(sorted_benefits) + 1)
            np.cumsum(sorted_benefits, out=cumsum_benefits[1:])
            
            max_available = len(sorted_benefits)
            
            # 3. Calculate Projected Value per Action
            action_costs = self.state['edge_costs'][valid_actions]
            
            # Fast Int Division:
            future_moves_arr = ((remaining_budget - action_costs + 1e-9) // self.min_edge_cost).astype(int)
            
            # Fully Vectorized Logic replacing Python Loop
            # Get ranks for all valid actions using the lookup table DIRECTLY!
            action_ranks = rank_lookup[valid_actions]

            # Create logical mask: Is the action considered part of the "top n"?
            # Condition: It has a valid rank (> -1) AND its rank (0-indexed) is less than n (count)
            in_top_set_mask = (action_ranks != -1) & (action_ranks < future_moves_arr)
            
            # Case A: Action IS in the top N set.
            # We sum the top (n+1) items, then subtract the specific action's value.
            # np.clip to ensure we don't exceed available items
            idxs_in = np.clip(future_moves_arr + 1, 0, max_available)
            
            # Use maximum(rank, 0) to avoid -1 indexing (safe because masked out later if rank is -1)
            safe_ranks = np.maximum(action_ranks, 0)
            vals_in = cumsum_benefits[idxs_in] - sorted_benefits[safe_ranks]
            
            # Case B: Action IS NOT in the top N set.
            # We simply sum the top n items.
            idxs_out = np.clip(future_moves_arr, 0, max_available)
            vals_out = cumsum_benefits[idxs_out]
            
            # Select value based on mask
            projected_values = np.where(in_top_set_mask, vals_in, vals_out)
            
        return projected_values

    def calculate_action_heuristics(self, valid_actions, flows, remaining_budget, include_projection=True, jitter=False, projection_uses_flow=False):
        """
        Calculate heuristic values for a batch of actions.
        Returns array of heuristic values aligned with valid_actions.
        """
        probs = self.state['edge_interdiction_probability'][valid_actions]
        
        projected_values = np.zeros(len(valid_actions))
        current_flow_vals = np.zeros(len(valid_actions))
        
        # Jitter logic
        if jitter and hasattr(self, 'flow_histories') and remaining_budget > 5:
            flow_histories = self.flow_histories
            
            # Compute flow values across all histories for all valid actions
            # Shape: (len(valid_actions), len(flow_histories))
            all_flow_vals = np.zeros((len(valid_actions), len(flow_histories)))
            
            for i, hist_flows in enumerate(flow_histories):
                if isinstance(hist_flows, np.ndarray):
                    all_flow_vals[:, i] = hist_flows[valid_actions]
                else:
                    all_flow_vals[:, i] = np.array([
                        hist_flows.get(self.both_edges[a], 0) + hist_flows.get((self.both_edges[a][1], self.both_edges[a][0]), 0)
                        for a in valid_actions
                    ])
            
            # Find the history index where each action's flow is minimized
            min_indices = np.argmin(all_flow_vals, axis=1)
            
            # Extract the minimized flow values
            current_flow_vals = all_flow_vals[np.arange(len(valid_actions)), min_indices]
            
            # Projections for jitter are based on capacity * probability (same as jitter=False, projection_uses_flow=False)
            if include_projection:
                projected_values = self._calculate_projections(
                    valid_actions, flows, remaining_budget, projection_uses_flow=False
                )
                        
        elif not projection_uses_flow:
            current_flow_vals = self.state['edge_capacity'][valid_actions]
            if include_projection:
                projected_values = self._calculate_projections(
                    valid_actions, flows, remaining_budget, projection_uses_flow
                )
        else:
            # Standard single-flow logic
            if isinstance(flows, np.ndarray):
                current_flow_vals = flows[valid_actions]
            elif hasattr(self, 'cached_flow_array') and flows is getattr(self, 'reference_flows', None):
                current_flow_vals = self.cached_flow_array[valid_actions].sum(axis=1)
            else:
                current_flow_vals = np.array([
                    flows.get(self.both_edges[a], 0) + flows.get((self.both_edges[a][1], self.both_edges[a][0]), 0)
                    for a in valid_actions
                ])
                
            if include_projection:
                projected_values = self._calculate_projections(
                    valid_actions, flows, remaining_budget, projection_uses_flow
                )
        
        heuristics = (probs * current_flow_vals) + projected_values
        
        return heuristics
    
    def solve_heuristic_interdiction(self):
        """
        Executes a Greedy Max-Flow Heuristic solver on the current environment state.
        
        The heuristic step:
        1. Calculate stochastic/deterministic flow on edges.
        2. Calculate heuristic score (Flow * Probability).
        3. Select best valid edge.
        4. Step environment.
        5. Repeat until done.
        
        Returns:
            final_obj (float): Final objective value.
            actions_taken (list): List of edge tuples interdicted.
        """
        actions_taken = []
        interdictable_edges_list = list(self.both_edges)
        
        done = False
        steps = 0
        max_steps = int(self.state['budget'][0] * 3) + 20 # Safety margin

        while not done and steps < max_steps:
             steps += 1
             
             # Get valid action mask
             action_mask = self.mask_fn()
             valid_actions = np.where(action_mask[:self.num_interdictable] == 1)[0]
             
             # Check if no valid actions remain
             if len(valid_actions) == 0:
                 # Select "do nothing" action
                 action = self.num_interdictable 
                 obs, reward, done, _, _ = self.step(action)
                 break
             
             # Get expected edge flows
             if self.deterministic_outcomes:
                 # Deterministic case: solve max flow directly
                 _, flows = self.solve_max_flow(routing_assumption=self.attacker_strategy)
             else:
                 # Stochastic case: get expected flows from stochastic calculation
                 _, flows = self._calculate_stochastic_objective_and_flow(
                     strategy_type=self.attacker_strategy, 
                     return_full_flows=True
                 )
             
             masked_values = np.full(self.num_interdictable, -np.inf)
             
             # Select the valid action with maximum (expected flow * interdiction probability)
             current_heuristics = self.calculate_action_heuristics(valid_actions, flows, self.state['budget'][0], include_projection=False)
             masked_values[valid_actions] = current_heuristics

             if self.attacker_strategy == "zero_sum":
                 pass

             elif self.attacker_strategy == "isolate":
                  pass

             elif self.attacker_strategy == "canalize":
                  # 1. Identify Target Nodes (Middle node of canalize objective)
                  canalize_obj = self.state['canalize_objective'][:self.num_both_edges]
                  obj_edges_indices = np.where(canalize_obj == 1)[0]
                  
                  path_edges = [self.both_edges[idx] for idx in obj_edges_indices]
                  
                  node_counts = {}
                  for u, v in path_edges:
                      node_counts[u] = node_counts.get(u, 0) + 1
                      node_counts[v] = node_counts.get(v, 0) + 1
                      
                  internal_nodes = [n for n, c in node_counts.items() if c >= 2]
                  
                  if not internal_nodes:
                      target_nodes = list(set([u for u,v in path_edges] + [v for u,v in path_edges]))
                  else:
                      internal_nodes.sort()
                      mid_idx = len(internal_nodes) // 2
                      target_nodes = [internal_nodes[mid_idx]]
                  
                  target_nodes_set = set(target_nodes)

                  # 2. Identify Target Edges (Connect to middle node)
                  target_set_actions = []
                  for act in valid_actions:
                      if canalize_obj[act] == 1:
                          continue
                      u, v = self.both_edges[act]
                      if u in target_nodes_set or v in target_nodes_set:
                          target_set_actions.append(act)
                          
                  best_action = -1
                  max_flow = -1
                  
                  # Priority 1: Connects to Middle Node
                  if target_set_actions:
                      for act in target_set_actions:
                          f = masked_values[act]
                          if f > max_flow:
                              max_flow = f
                              best_action = act
                  
                  # Priority 2: Connects to First Node (Flow Away)
                  if best_action == -1:
                      endpoints = [n for n, c in node_counts.items() if c == 1]
                      start_node = None
                      for ep in endpoints:
                          if ep not in self.sink_nodes and ep not in self.super_sink_nodes:
                              start_node = ep
                              break
                      if start_node is None and endpoints:
                          start_node = endpoints[0]

                      if start_node:
                          is_start_interdicted = False
                          if start_node in self.edge_groups:
                              for edge in self.edge_groups[start_node]['out'] + self.edge_groups[start_node]['in']:
                                  idx = self.edge_to_index.get(edge)
                                  if idx is not None and (0 <= idx < len(self.state['edge_interdicted'])) and self.state['edge_interdicted'][idx] >= 1:
                                      is_start_interdicted = True
                                      break
                          
                          if not is_start_interdicted:
                              start_node_actions = []
                              for act in valid_actions:
                                  if canalize_obj[act] == 1: continue
                                  u, v = self.both_edges[act]
                                  if u == start_node or v == start_node:
                                      start_node_actions.append(act)
                              
                              if start_node_actions:
                                  max_start_flow = -1
                                  for act in start_node_actions:
                                      f = masked_values[act]
                                      if f > max_start_flow:
                                          max_start_flow = f
                                          best_action = act

                  # Priority 3: Zero Sum behavior
                  if best_action == -1:
                      pass
                  else:
                      masked_values[best_action] = float('inf')

             elif self.attacker_strategy == "divert":
                  # Placeholder - default to max flow heuristic for now
                  pass
             
             action = int(np.argmax(masked_values))
             
             # Step the environment with selected action
             obs, reward, done, _, _ = self.step(action)

             # Tally actions
             if action < self.num_interdictable:
                 action_key = interdictable_edges_list[action]
                 actions_taken.append(action_key)
                 
             if done:
                 break
        
        # Determine final objective value based on strategy
        if self.attacker_strategy == "zero_sum":
            objVal, _ = self._compute_objective_and_flows()
        elif self.attacker_strategy == "isolate":
            objVal, _ = self._calculate_isolate_objective_and_flows()
        elif self.attacker_strategy == "canalize":
            objVal, _ = self._calculate_canalize_objective_and_flows()
        elif self.attacker_strategy == "divert":
            objVal, _ = self._calculate_divert_objective_and_flows()
        else:
            objVal, _ = self._compute_objective_and_flows()

        return objVal, actions_taken

    def solve_min_cut_heuristic(self, return_details=False):
        """
        Executes a Min-Cut Heuristic solver on the current environment state.
        
        Variations:
        - Zero Sum: Min-cut separates Super Source from Super Sink.
        - Isolate: Min-cut separates Super Source from target nodes in isolate_objective.

        Args:
            return_details (bool): If True, returns (objVal, actions_taken, cut_set, node_partitions),
                                   where cut_set is the set of edges in the min-cut (y=1)
                                   and node_partitions is a dict of node -> 0/1 (p values).
                                   If False, returns (objVal, actions_taken).
        """
        actions_taken = []
        num_interdictable = len(self.interdictable_edges)
        interdictable_set = set(self.interdictable_edges)
        
        # Variables to store extra details if requested
        cut_set = set()
        node_partitions = {}

        if self.attacker_strategy in ["zero_sum", "isolate"]:
            # 1. Compute Min Cut using Gurobi
            cut_model = grb.Model("MinCut_Heuristic", env=self.GUROBI_ENV)
            cut_model.setParam('OutputFlag', 0)
            
            # Nodes: 0 if on source side, 1 if on sink side
            p = cut_model.addVars(self.nodes, vtype=grb.GRB.BINARY, name="p")
            # Edges: 1 if edge is in the cut
            y = cut_model.addVars(self.both_edges, vtype=grb.GRB.BINARY, name="y")
            z = cut_model.addVars(self.both_edges, vtype=grb.GRB.BINARY, name="z")
            
            # Boundary conditions
            cut_model.addConstr(p[self.super_source_nodes[0]] == 0)
            
            target_nodes = set()
            super_sinks = set(self.super_sink_nodes)
            if self.attacker_strategy == "zero_sum":
                cut_model.addConstr(p[self.super_sink_nodes[0]] == 1)
            elif self.attacker_strategy == "isolate":
                target_indices = np.where(self.state['isolate_objective'][:self.num_both_edges] == 1)[0]
                target_nodes = set([self.both_edges[i][1] for i in target_indices])
                for node in target_nodes:
                    cut_model.addConstr(p[node] == 1)
            
            # Constraint: y_uv >= p_v - p_u (Standard min-cut formulation)
            # We use absolute difference for the symmetric capacity assumption
            for idx, edge in enumerate(self.both_edges):
                u, v = edge
                
                # USER REQUIREMENT: Skip constraints for isolate strategy on target-sink edges
                skip_partition_const = False
                if self.attacker_strategy == "isolate":
                    if (u in target_nodes and v in super_sinks) or (v in target_nodes and u in super_sinks):
                        skip_partition_const = True
                
                if not skip_partition_const:
                    cut_model.addConstr(y[edge] >= p[v] - p[u])
                    cut_model.addConstr(y[edge] >= p[u] - p[v])
                
                cut_model.addConstr(z[edge] <= y[edge])

                # USER REQUIREMENT: Non-interdictable OR Zero Success Probability edges cannot be in the cut
                prob = self.state['edge_interdiction_probability'][idx]
                if edge not in interdictable_set or prob <= 1e-9:
                    y[edge].UB = 0

            # 1. Budget constraint: sum of (z[edge] * cost) <= current state budget
            cut_model.addConstr(
                grb.quicksum(self.state['edge_costs'][idx] * z[edge] for idx, edge in enumerate(self.both_edges))
                <= self.state['budget'][0], name="budget"
            )
            
            # 2. Objective: Minimize (Capacity * y) - (Capacity * prob * z)
            cut_model.setObjective(
                grb.quicksum(self.state['edge_capacity'][idx] * y[edge] - 
                             (self.state['edge_capacity'][idx] * self.state['edge_interdiction_probability'][idx] * z[edge])
                             for idx, edge in enumerate(self.both_edges)),
                grb.GRB.MINIMIZE
            )
            
            cut_model.optimize()
            
            # Interdict edges where z[edge].X is 1 - Batch Update
            if cut_model.status == grb.GRB.OPTIMAL:
                if return_details:
                     # Capture all cut edges (y=1)
                     for edge in self.both_edges:
                         if y[edge].X > 0.5:
                             cut_set.add(edge)
                     # Capture partitions (p values)
                     for node in self.nodes:
                         node_partitions[node] = int(round(p[node].X))

                for idx, edge in enumerate(self.both_edges):
                    if z[edge].X > 0.5:
                        action_idx = self.edge_to_index.get(edge)
                        if action_idx is not None:
                            # Update state directly (bypass step/mask_fn for efficiency as requested)
                            self.state['edge_interdicted'][action_idx] += 1
                            self.state['budget'][0] -= self.state['edge_costs'][action_idx]
                            actions_taken.append(edge)
            
        elif self.attacker_strategy == "canalize":
            # Identify canalize objective edges
            target_indices = np.where(self.state['canalize_objective'][:self.num_both_edges] == 1)[0]
            target_indices_set = set(target_indices)

            # Use _extract_directed_path_edges to get ordered path and nodes
            path_edges = self._extract_directed_path_edges('canalize_objective')
            
            # derive ordered node sequence
            node_seq = [e[0] for e in path_edges] + [path_edges[-1][1]]
            start_node = node_seq[0]
            end_node = node_seq[-1]
            intermediate_nodes = set(node_seq[1:-1])

            # Edges of interest: incident to intermediate nodes but not part of canalize objective
            edges_of_interest_idxs = set()
            for node in intermediate_nodes:
                connected = self.edge_groups[node].get('in', []) + self.edge_groups[node].get('out', [])
                for edge in connected:
                    idx = self.edge_to_index.get(edge)
                    # Also try reversed orientation if direct lookup fails
                    if idx is None:
                        rev = (edge[1], edge[0])
                        idx = self.edge_to_index.get(rev)
                    if idx is not None and idx not in target_indices_set:
                        edges_of_interest_idxs.add(idx)

            # Build a single min-cut model that forces an edge from canalize objective into the cut
            cut_model = grb.Model("MinCut_Canalize_Targeted", env=self.GUROBI_ENV)
            cut_model.setParam('OutputFlag', 0)

            p = cut_model.addVars(self.nodes, vtype=grb.GRB.BINARY, name="p")
            y = cut_model.addVars(self.both_edges, vtype=grb.GRB.BINARY, name="y")
            z = cut_model.addVars(self.both_edges, vtype=grb.GRB.BINARY, name="z")

            # Use start/end nodes from extracted path
            cut_model.addConstr(p[start_node] == 0) #start_node
            cut_model.addConstr(p[1] == 0) #start_node
            cut_model.addConstr(p[end_node] == 1) #end_node
            cut_model.addConstr(p[250] == 1) #end_node
           
            # Force edges of interest to be interdicted (z==1).
            for idx in edges_of_interest_idxs:
                edge = self.both_edges[idx]
                prob = self.state['edge_interdiction_probability'][idx]
                # Force interdiction (z==1) when possible. Do NOT force y==1 for edges of interest.
                if edge in interdictable_set and prob > 1e-9:
                    cut_model.addConstr(z[edge] == 1)

            # Ensure at least one canalize objective edge is in the cut (y==1) and none are interdicted (z==0).
            for idx in target_indices:
                edge = self.both_edges[idx]
                cut_model.addConstr(z[edge] == 0)
            cut_model.addConstr(grb.quicksum(y[self.both_edges[idx]] for idx in target_indices) >= 1)
            
            # Standard min-cut constraints
            for idx, edge in enumerate(self.both_edges):
                u, v = edge
                cut_model.addConstr(y[edge] >= p[v] - p[u])
                cut_model.addConstr(y[edge] >= p[u] - p[v])

                # Only enforce z <= y for edges that are NOT in edges_of_interest
                if idx not in edges_of_interest_idxs:
                    cut_model.addConstr(z[edge] <= y[edge])

                # If edge not interdictable or prob==0, forbid z
                prob = self.state['edge_interdiction_probability'][idx]
                if edge not in interdictable_set or prob <= 1e-9:
                    z[edge].UB = 0

            # Budget constraint: exclude the canalize objective edges from costing
            cost_terms = []
            for idx, edge in enumerate(self.both_edges):
                if idx in target_indices_set:
                    # skip cost 
                    continue
                cost_terms.append(self.state['edge_costs'][idx] * z[edge])

            cut_model.addConstr(grb.quicksum(cost_terms) <= self.state['budget'][0], name="budget")

            # Objective: same as zero_sum: minimize capacity*y - expected reduction from z
            cut_model.setObjective(
                grb.quicksum(self.state['edge_capacity'][idx] * y[edge] -
                             (self.state['edge_capacity'][idx] * self.state['edge_interdiction_probability'][idx] * z[edge])
                             for idx, edge in enumerate(self.both_edges)),
                grb.GRB.MINIMIZE
            )

            cut_model.optimize()

            # Batch apply interdictions from z (but never interdicted canalize objective edges by constraint)
            if cut_model.status == grb.GRB.OPTIMAL:
                if return_details:
                     # Capture all cut edges (y=1)
                     for edge in self.both_edges:
                         if y[edge].X > 0.5:
                             cut_set.add(edge)
                     # Capture partitions (p values)
                     for node in self.nodes:
                         node_partitions[node] = int(round(p[node].X))

                for idx, edge in enumerate(self.both_edges):
                    # Skip canalize objective edges explicitly
                    if idx in target_indices_set:
                        continue
                    if z[edge].X > 0.5:
                        action_idx = self.edge_to_index.get(edge)
                        if action_idx is not None:
                            # Update state
                            self.state['edge_interdicted'][action_idx] += 1
                            self.state['budget'][0] -= self.state['edge_costs'][action_idx]
                            actions_taken.append(edge)

        elif self.attacker_strategy == "divert":  #PICKUP HERE!!!
            # Identify canalize objective edges
            target_to_indices = np.where(self.state['divert_to_objective'][:self.num_both_edges] == 1)[0]
            target_from_indices = np.where(self.state['divert_from_objective'][:self.num_both_edges] == 1)[0]
            target_to_indices_set = set(target_to_indices)
            target_from_indices_set = set(target_from_indices)

            # Use _extract_directed_path_edges to get ordered path and nodes
            path_edges_to = self._extract_directed_path_edges('divert_to_objective')
            path_edges_from = self._extract_directed_path_edges('divert_from_objective')
            
            # derive ordered node sequence
            node_seq_to = [e[0] for e in path_edges_to] + [path_edges_to[-1][1]]
            start_node = node_seq_to[0]
            end_node = node_seq_to[-1]
            intermediate_nodes_to = set(node_seq_to[1:-1])

            # Edges of interest: incident to intermediate nodes but not part of divert_to objective
            edges_of_interest_idxs = set()
            for node in intermediate_nodes_to:
                connected = self.edge_groups[node].get('in', []) + self.edge_groups[node].get('out', [])
                for edge in connected:
                    idx = self.edge_to_index.get(edge)
                    # Also try reversed orientation if direct lookup fails
                    if idx is None:
                        rev = (edge[1], edge[0])
                        idx = self.edge_to_index.get(rev)
                    if idx is not None and idx not in target_to_indices_set:
                        edges_of_interest_idxs.add(idx)

            # Build a single min-cut model that forces an edge from divert_to objective and divert_from objective into the cut
            cut_model = grb.Model("MinCut_Divert_Targeted", env=self.GUROBI_ENV)
            cut_model.setParam('OutputFlag', 0)

            p = cut_model.addVars(self.nodes, vtype=grb.GRB.BINARY, name="p")
            y = cut_model.addVars(self.both_edges, vtype=grb.GRB.BINARY, name="y")
            z = cut_model.addVars(self.both_edges, vtype=grb.GRB.BINARY, name="z")

            # Use start/end nodes from extracted path
            cut_model.addConstr(p[start_node] == 0) #start_node
            cut_model.addConstr(p[1] == 0) #start_node
            cut_model.addConstr(p[end_node] == 1) #end_node
            cut_model.addConstr(p[250] == 1) #end_node
           
            # Force edges of interest to be interdicted (z==1).
            for idx in edges_of_interest_idxs:
                edge = self.both_edges[idx]
                prob = self.state['edge_interdiction_probability'][idx]
                # Force interdiction (z==1) when possible. Do NOT force y==1 for edges of interest.
                if edge in interdictable_set and prob > 1e-9:
                    cut_model.addConstr(z[edge] == 1)

            # Ensure at least one divert_to objective edge is in the cut (y==1) and none are interdicted (z==0).
            for idx in target_to_indices:
                edge = self.both_edges[idx]
                cut_model.addConstr(z[edge] == 0)
            cut_model.addConstr(grb.quicksum(y[self.both_edges[idx]] for idx in target_to_indices) >= 1)
            
            # Standard min-cut constraints
            for idx, edge in enumerate(self.both_edges):
                u, v = edge
                cut_model.addConstr(y[edge] >= p[v] - p[u])
                cut_model.addConstr(y[edge] >= p[u] - p[v])

                # Only enforce z <= y for edges that are NOT in edges_of_interest
                if idx not in edges_of_interest_idxs:
                    cut_model.addConstr(z[edge] <= y[edge])

                # If edge not interdictable or prob==0, forbid z
                prob = self.state['edge_interdiction_probability'][idx]
                if edge not in interdictable_set or prob <= 1e-9:
                    z[edge].UB = 0

            # Budget constraint: exclude the divert_to objective edges from costing
            cost_terms = []
            for idx, edge in enumerate(self.both_edges):
                if idx in target_to_indices_set:
                    # skip cost 
                    continue
                cost_terms.append(self.state['edge_costs'][idx] * z[edge])

            cut_model.addConstr(grb.quicksum(cost_terms) <= self.state['budget'][0], name="budget")

            # Objective: same as zero_sum: minimize capacity*y - expected reduction from z
            cut_model.setObjective(
                grb.quicksum(self.state['edge_capacity'][idx] * y[edge] -
                             (self.state['edge_capacity'][idx] * self.state['edge_interdiction_probability'][idx] * z[edge])
                             for idx, edge in enumerate(self.both_edges)),
                grb.GRB.MINIMIZE
            )

            cut_model.optimize()

            # Batch apply interdictions from z (but never interdicted canalize objective edges by constraint)
            if cut_model.status == grb.GRB.OPTIMAL:
                if return_details:
                     # Capture all cut edges (y=1)
                     for edge in self.both_edges:
                         if y[edge].X > 0.5:
                             cut_set.add(edge)
                     # Capture partitions (p values)
                     for node in self.nodes:
                         node_partitions[node] = int(round(p[node].X))

                for idx, edge in enumerate(self.both_edges):
                    # Skip divert_to objective edges explicitly
                    if idx in target_to_indices_set:
                        continue
                    if z[edge].X > 0.5:
                        action_idx = self.edge_to_index.get(edge)
                        if action_idx is not None:
                            # Update state
                            self.state['edge_interdicted'][action_idx] += 1
                            self.state['budget'][0] -= self.state['edge_costs'][action_idx]
                            actions_taken.append(edge)
            
        # Determine final objective value based on strategy
        if self.attacker_strategy == "zero_sum":
            objVal, _ = self._compute_objective_and_flows()
        elif self.attacker_strategy == "isolate":
            objVal, _ = self._calculate_isolate_objective_and_flows()
        elif self.attacker_strategy == "canalize":
            objVal, _ = self._calculate_canalize_objective_and_flows()
        elif self.attacker_strategy == "divert":
            objVal, _ = self._calculate_divert_objective_and_flows()
        else:
            objVal, _ = self._compute_objective_and_flows()

        if return_details:
            return objVal, actions_taken, cut_set, node_partitions
        else:
            return objVal, actions_taken

    # --- Re-add solve_backward_induction_ray method to Mixin ---
    
    def solve_backward_induction_ray(self, verbose=False, n_workers=4, worker_depth=None, ray_address=None, enable_memoization=True, enable_outcome_caching=True, enable_alpha_pruning=True, rl_model_path=None, time_limit=3600, reduce_flow=False, parallel_expansion=False, jitter=False, projection_uses_flow=False, objective_tolerance=1e-5):
        """
        Parallelized backward induction using Ray with Adaptive Frontier Expansion.
        """
        # Import locally to ensure availability in all paths and avoid scope issues
        import copy, numpy as np, ray as _ray, time

        original_state = copy.deepcopy(self.state)
        original_enable_outcome_caching = getattr(self, 'enable_outcome_caching', True)
        original_reduce_flow = getattr(self, 'reduce_flow', False)
        original_local_outcome_cache = copy.deepcopy(getattr(self, 'local_outcome_cache', {}))
        original_outcome_memo_actors = getattr(self, 'outcome_memo_actors', None)
        original_reference_flows = copy.deepcopy(getattr(self, 'reference_flows', None))
        original_reference_flows_dict = copy.deepcopy(getattr(self, 'reference_flows_dict', None))
        original_reference_start_flow = getattr(self, 'reference_start_flow', None)
        original_reference_start_flows = copy.deepcopy(getattr(self, 'reference_start_flows', None))
        original_reference_obj = getattr(self, 'reference_obj', None)
        original_reference_budget = getattr(self, 'reference_budget', None)
        original_last_obj = getattr(self, 'last_obj', None)
        original_canalize_norm_factor = getattr(self, 'canalize_norm_factor', None)
        original_max_canalize_objective = getattr(self, 'max_canalize_objective', None)
        original_max_divert_objective = getattr(self, 'max_divert_objective', None)
        original_cached_flow_array = copy.deepcopy(getattr(self, 'cached_flow_array', None))

        start_time = time.time()
        
        if jitter:
            enable_alpha_pruning = True
            reduce_flow = True

        # init ray if not already
        if n_workers > 0 and not ray.is_initialized():
            ray.init(address=ray_address, ignore_reinit_error=True)

        # Ensure clean Gurobi model state for determinism
        self._cleanup_models()

        # Propagate caching flag to driver for heuristic usage
        self.enable_outcome_caching = enable_outcome_caching
        self.reduce_flow = reduce_flow
        if self.enable_outcome_caching:
             self.local_outcome_cache = {}

        # Create outcome memoization actor ONLY if stochastic
        outcome_memo_actors = []
        if not self.deterministic_outcomes and enable_outcome_caching:
            num_outcome_shards = min(10, n_workers) if n_workers > 0 else 1
            outcome_memo_actors = [SharedOutcomeMemoActor.remote() for _ in range(num_outcome_shards)]
            self.outcome_memo_actors = outcome_memo_actors

        initial_alpha = -float('inf')
        initial_alpha_actions = []
        original_initial_alpha = -float('inf')

        if rl_model_path:
            if verbose:
                print(f"Loading RL model from {rl_model_path} for initial alpha...")
            try:
                # Save state
                old_budget = self.state['budget'][0]
                old_interdicted = self.state['edge_interdicted'].copy()
                
                # Load Model
                agent = None
                if "MaskablePPO" in rl_model_path:
                    from sb3_contrib import MaskablePPO
                    agent = MaskablePPO.load(rl_model_path)
                else:
                    if verbose: print("Warning: Unknown model type in path. Trying PPO.")

                # Run Episode
                done = False
                while not done:
                    # Update flows for mask calculation
                    if self.attacker_strategy == "zero_sum":
                         _, self.reference_flows = self._compute_objective_and_flows()
                    elif self.attacker_strategy == 'canalize':
                         _, self.reference_flows = self._calculate_canalize_objective_and_flows()
                    elif self.attacker_strategy == 'isolate':
                         _, self.reference_flows = self._calculate_isolate_objective_and_flows()
                    elif self.attacker_strategy == 'divert':
                         _, self.reference_flows = self._calculate_divert_objective_and_flows()
                    
                    self._cache_flow_array()

                    # Get observation
                    # Assuming observation dict matches state or is part of it. 
                    # The env_TA.py usually returns the state dict as obs.
                    obs = self.state 
                    
                    if "MaskablePPO" in rl_model_path:
                        action_masks = self.mask_fn()
                        action, _ = agent.predict(obs, action_masks=action_masks, deterministic=True)
                    else:
                        action, _ = agent.predict(obs, deterministic=True)

                    # Step
                    # Since we are In the env, we can just call step() directly?
                    # self.step() might expect self to be a Gym Env.
                    # We are in a Mixin, but the class using it is CustomEnv(gym.Env). 
                    # So self.step(int(action)) should work.
                    
                    _, _, done, _, _ = self.step(int(action))
                    
                    if int(action) < self.num_interdictable:
                        initial_alpha_actions.append(self.both_edges[int(action)])

                # Get final objective
                if self.attacker_strategy == "zero_sum":
                    obj_val, _ = self._compute_objective_and_flows()
                    obj_val = -obj_val
                elif self.attacker_strategy == "isolate":
                    obj_val, _ = self._calculate_isolate_objective_and_flows()
                    obj_val = -obj_val
                elif self.attacker_strategy == "canalize":
                    obj_val, _ = self._calculate_canalize_objective_and_flows()
                elif self.attacker_strategy == "divert":
                    obj_val, _ = self._calculate_divert_objective_and_flows()
                else:
                    obj_val = -float('inf')

                initial_alpha = obj_val
                
                if verbose:
                    print(f"RL Model found initial alpha: {initial_alpha}")

            except Exception as e:
                if verbose:
                    print(f"RL Model execution failed: {e}. Falling back to heuristic.")
                rl_model_path = None # Trigger fallback below if needed
            finally:
                # Restore state
                self.state['budget'][0] = old_budget
                self.state['edge_interdicted'][:] = old_interdicted

        if enable_alpha_pruning and not rl_model_path:
            if verbose:
                print("Running heuristic (Min-Cut) for initial alpha...")
            
            # Save state
            old_budget = self.state['budget'][0]
            old_interdicted = self.state['edge_interdicted'].copy()
            
            try:
                # Use Min-Cut Heuristic to get a tight lower bound (initial alpha)
                heuristic_val, heuristic_actions = self.solve_min_cut_heuristic()
                
                # Convert heuristic value to "reward" space (maximize negative flow)
                if self.attacker_strategy in ["zero_sum", "isolate"]:
                    initial_alpha = -heuristic_val
                else:
                    initial_alpha = heuristic_val
                
                # Store actions for optimal sequence initialization
                initial_alpha_actions = heuristic_actions

            except Exception as e:
                if verbose:
                    print(f"Min-Cut Heuristic failed: {e}")
                initial_alpha = -float('inf')
            finally:
                if initial_alpha > -float('inf'):
                    # Small safety margin for floating point comparisons during pruning
                    initial_alpha -= 1e-4

                # Restore state
                self.state['budget'][0] = old_budget
                self.state['edge_interdicted'][:] = old_interdicted

                 # Recalculate flows for state restoration (fix for reference_flows error)
                try:
                    if self.attacker_strategy == "zero_sum":
                        _, self.reference_flows = self._compute_objective_and_flows()
                    elif self.attacker_strategy == 'canalize':
                        _, self.reference_flows = self._calculate_canalize_objective_and_flows()
                    elif self.attacker_strategy == 'isolate':
                        _, self.reference_flows = self._calculate_isolate_objective_and_flows()
                    elif self.attacker_strategy == 'divert':
                        _, self.reference_flows = self._calculate_divert_objective_and_flows()
                    
                    self._cache_flow_array()
                except Exception:
                    pass

            if verbose:
                print(f"Heuristic found initial alpha: {initial_alpha}")

        # SERIAL EXECUTION PATH
        if n_workers <= 0:
            # Propagate implementation flag to the env used by the serial process
            self.enable_outcome_caching = enable_outcome_caching

            if verbose:
                print(f"Running in SERIAL mode (Main Process). Memoization: {'ON' if enable_memoization else 'OFF'}")
            
            # Local Memoization
            memo_serial = {}
            
            # Calculate budget stats for progress bar
            max_budget = self.state['budget'][0]
            budget_levels = int(max_budget // self.min_edge_cost) if self.min_edge_cost > 0 else 1
            estimated_states = (int(self.num_both_edges) ** budget_levels) if budget_levels > 0 else 1
            
            from tqdm import tqdm
            pbar = tqdm(total=estimated_states, desc="DP States (Serial)", unit=" states", disable=not verbose)

            # Define recursive solver for serial execution
            best_incumbent_reward = initial_alpha
            best_incumbent_seq = [self.edge_to_index[e] for e in initial_alpha_actions]
            serial_pruned_count = 0
            serial_invalid_count = 0
            serial_memo_count = 0
            serial_base_count = 0

            def dp_serial(rem_budget, inter_state, d, alpha=-float('inf'), path_to_here=[]):
                nonlocal best_incumbent_reward, best_incumbent_seq, serial_pruned_count, serial_invalid_count, serial_memo_count, serial_base_count
                
                if time.time() - start_time > time_limit:
                    raise TimeoutError("Time limit exceeded")

                key = inter_state[:self.num_both_edges].tobytes()
                
                # Volume calc for this node's potential subtree
                current_volume = int(int(self.num_both_edges) ** max(0, budget_levels - d))

                if enable_memoization and key in memo_serial:
                    pbar.update(current_volume)
                    serial_memo_count += current_volume
                    return memo_serial[key]
                
                # Save state
                old_budget = self.state['budget'][0]
                old_interdicted = self.state['edge_interdicted'].copy()
                
                # Set state
                self.state['budget'][0] = rem_budget
                self.state['edge_interdicted'][:] = inter_state
                
                # Compute objective
                if self.attacker_strategy == "zero_sum":
                    val, flows = self._compute_objective_and_flows()
                    val = -val # Attacker maximizes negative flow (minimizes flow)
                elif self.attacker_strategy == 'canalize':
                    val, flows = self._calculate_canalize_objective_and_flows()
                elif self.attacker_strategy == 'isolate':
                    val, flows = self._calculate_isolate_objective_and_flows()
                    val = -val
                elif self.attacker_strategy == 'divert':
                    val, flows = self._calculate_divert_objective_and_flows()
                else:
                    val = -float('inf')
                    flows = {}
                
                # Update incumbent if we found a better terminal value (or stopping here)
                if val > best_incumbent_reward + objective_tolerance:
                    best_incumbent_reward = val
                    best_incumbent_seq = list(path_to_here)
                elif abs(val - best_incumbent_reward) <= objective_tolerance:
                    current_cost = sum(self.state['edge_costs'][a] for a in best_incumbent_seq)
                    new_cost = sum(self.state['edge_costs'][a] for a in path_to_here)
                    if new_cost < current_cost:
                        best_incumbent_seq = list(path_to_here)

                self.reference_flows = flows
                self._cache_flow_array()
                
                # Base case
                if rem_budget < self.min_edge_cost or (worker_depth is not None and d >= worker_depth):
                    # Restore
                    self.state['budget'][0] = old_budget
                    self.state['edge_interdicted'][:] = old_interdicted
                    
                    if enable_memoization:
                        memo_serial[key] = (val, [])
                    pbar.update(current_volume)
                    serial_base_count += current_volume
                    return val, []

                # Get actions
                action_mask = self.mask_fn()
                
                # Restore state immediately after mask calculation to keep environment clean
                self.state['budget'][0] = old_budget
                self.state['edge_interdicted'][:] = old_interdicted
                
                valid_actions = np.where(action_mask[:self.num_both_edges] == 1)[0]
                
                # Account for branches pruned by invalid actions
                num_invalid = int(self.num_both_edges) - len(valid_actions)
                if num_invalid > 0:
                    child_volume = int(int(self.num_both_edges) ** max(0, budget_levels - (d + 1)))
                    inv_vol = num_invalid * child_volume
                    pbar.update(inv_vol)
                    serial_invalid_count += inv_vol

                if len(valid_actions) == 0:
                    if enable_memoization:
                        memo_serial[key] = (val, [])
                    return val, []
                    
                best_reward = val
                best_seq = []
                
                alpha = max(alpha, best_reward)
                
                # Apply Heuristics (Same as in Worker)
                if enable_alpha_pruning:
                    # Set temporary state for accurate heuristic projection benefit calculation
                    self.state['budget'][0] = rem_budget
                    self.state['edge_interdicted'][:] = inter_state
                    
                    heuristics = self.calculate_action_heuristics(valid_actions, flows, rem_budget, projection_uses_flow=projection_uses_flow)
                    
                    # Restore state immediately
                    self.state['budget'][0] = old_budget
                    self.state['edge_interdicted'][:] = old_interdicted
                    
                    sorted_indices = np.argsort(-heuristics)
                    valid_actions = valid_actions[sorted_indices]
                    heuristics = heuristics[sorted_indices]

                for i, action in enumerate(valid_actions):
                    # Pruning
                    if enable_alpha_pruning:
                         if val + heuristics[i] < alpha- objective_tolerance:
                             skipped_actions = len(valid_actions) - i
                             child_volume = int(int(self.num_both_edges) ** max(0, budget_levels - (d + 1)))
                             pruned_vol = skipped_actions * child_volume
                             pbar.update(pruned_vol)
                             serial_pruned_count += pruned_vol
                             break

                    inter_state[action] += 1
                    new_budget = rem_budget - self.state['edge_costs'][action]
                    
                    path_to_here.append(action)
                    fut_reward, fut_seq = dp_serial(new_budget, inter_state, d + 1, alpha, path_to_here)
                    path_to_here.pop()
                    
                    inter_state[action] -= 1
                    
                    if fut_reward > best_reward + objective_tolerance:
                        best_reward = fut_reward
                        best_seq = [action] + fut_seq
                        alpha = max(alpha, best_reward)
                    elif abs(fut_reward - best_reward) <= objective_tolerance:
                        # Tie-breaking within tolerance: favor sequence with the minimal total cost (highest depth)
                        current_cost = sum(self.state['edge_costs'][a] for a in best_seq)
                        new_cost = self.state['edge_costs'][action] + sum(self.state['edge_costs'][a] for a in fut_seq)
                        if new_cost < current_cost:
                            best_seq = [action] + fut_seq
                        
                if enable_memoization:
                    memo_serial[key] = (best_reward, best_seq)
                return best_reward, best_seq

            # Run Serial
            t0 = time.time()
            initial_interdicted = self.state['edge_interdicted'].copy()
            initial_budget = self.state['budget'][0]
            
            try:
                opt_reward, opt_seq = dp_serial(initial_budget, initial_interdicted, 0, alpha=initial_alpha, path_to_here=[])
            except TimeoutError:
                if verbose:
                    print(f"Serial execution timed out after {time.time() - t0:.2f}s. Returning best solution found.")
                opt_reward = best_incumbent_reward
                opt_seq = best_incumbent_seq

            pbar.close()

            total_states_evaluated = estimated_states
            if total_states_evaluated > 0:
                pruned_pct = (serial_pruned_count / total_states_evaluated * 100)
                invalid_pct = (serial_invalid_count / total_states_evaluated * 100)
                memo_pct = (serial_memo_count / total_states_evaluated * 100)
                base_pct = (serial_base_count / total_states_evaluated * 100)
                
                print(f"--- DP State Traversal Breakdown ---")
                print(f"Alpha Pruning States Dropped:   {serial_pruned_count:,} ({pruned_pct:.2f}%)")
                print(f"Invalid Actions States Dropped: {serial_invalid_count:,} ({invalid_pct:.2f}%)")
                print(f"Memoization Cache Hits:         {serial_memo_count:,} ({memo_pct:.2f}%)")
                print(f"Leaves Actually Visited:        {serial_base_count:,} ({base_pct:.2f}%)")
                print(f"Total States (Estimated):       {total_states_evaluated:,}")
                print(f"------------------------------------")

            if verbose:
                print(f"Serial execution completed in {time.time() - t0:.2f}s")
                
            if opt_reward <= initial_alpha + 1.1e-4:
                print("Note: The initial alpha (from heuristic or RL) was not beaten during backward induction.")

            optimal_actions = [self.both_edges[idx] for idx in opt_seq]
            
            # Match behavior of parallel implementation and Gurobi
            if self.attacker_strategy in ("zero_sum", "isolate"):
                opt_reward = -opt_reward
                
            return opt_reward, optimal_actions


        # snapshot state to send to workers
        state_snapshot = copy.deepcopy(self.state)
        seed = getattr(self, 'seed', None)

        # Initialize actors to None for safe cleanup
        progress_actor = None
        memo_actors = []
        alpha_actor = None
        workers = []

        try:
            # create actors
            progress_actor = _ProgressActor.remote()
            # SHARDED MEMOIZATION
            # Create multiple memo actors to reduce lock contention
            # Using n_workers shards ensures high throughput
            num_memo_shards = min(2, n_workers) 
            
            if enable_memoization:
                memo_actors = [_SharedMemoActor.remote() for _ in range(num_memo_shards)]
            else:
                memo_actors = []

            # outcome_memo_actors already created before heuristic

            # Create alpha actor
            # When using heuristics, the initial_alpha is a lower bound on the optimal value.
            # We should subtract a small epsilon (or larger buffer) to ensure we don't prune branches that are exactly equal 
            # to this initial value due to floating point noise.
            initial_alpha_indices = [self.edge_to_index[e] for e in initial_alpha_actions]
            alpha_actor = _SharedAlphaActor.remote(initial_alpha, initial_alpha_indices)
            
            best_incumbent_val = initial_alpha
            best_incumbent_seq = initial_alpha_indices
            
            max_budget = self.state['budget'][0]
            budget_levels = int(max_budget // self.min_edge_cost) if self.min_edge_cost > 0 else 1

            # Optimization: Put large static data in Ray object store once to avoid repeated serialization
            nodes_ref = ray.put(self.nodes)
            edges_ref = ray.put(self.edges_reset)
            state_ref = ray.put(state_snapshot)

            env_refs = {}
            for attr in ['reference_flows_dict', 'reference_start_flow', 'canalize_norm_factor', 'reference_obj', 'reference_flows', 'max_canalize_objective', 'reference_start_flows', 'max_divert_objective']:
                if hasattr(self, attr):
                    env_refs[attr] = getattr(self, attr)

            workers = [
                _RemoteEnvWorker.remote(
                    nodes_ref,
                    edges_ref,
                    seed,
                    state_ref,
                    self.attacker_strategy,
                    self.min_edge_cost,
                    self.num_both_edges,
                    self.deterministic_outcomes,
                    self.multiple_interdiction_attempts,
                    progress_actor=progress_actor,
                    memo_actors=memo_actors, # Pass list of actors
                    budget_levels=budget_levels,
                    progress_granularity=2000,
                    jitter=jitter,
                    max_depth_inner=100,
                    outcome_memo_actors=outcome_memo_actors,
                    alpha_actor=alpha_actor,
                    enable_outcome_caching=enable_outcome_caching,
                    enable_alpha_pruning=enable_alpha_pruning,
                    sample_size=self.SAMPLE_SIZE,
                    reduce_flow=reduce_flow,
                    projection_uses_flow=projection_uses_flow,
                    env_references=env_refs
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
                        current, _, _, _, _ = ray.get(progress_actor.get_count.remote(), timeout=1)
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
                    self.stopping_value = None
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
            tasks_to_solve = [] # Nodes ready to be sent to workers

            if parallel_expansion:
                if verbose:
                    print("Running Parallel Frontier Expansion...")
                
                node_registry = {id(root_node): root_node} # Map ID -> Object
                
                # Running futures: future_ref -> valid_node_ids_in_batch
                expansion_futures = {} 
                idle_workers_expansion = list(workers)
                
                BATCH_SIZE = 10 # Batch size per worker for vectorization benefits
                MAX_EXPANSION_DEPTH = 20
                
                while frontier or expansion_futures:
                    # Check timeout
                    if time.time() - start_time > time_limit:
                        if verbose: print("Expansion phase timed out.")
                        tasks_to_solve.extend(frontier)
                        frontier = []
                        break

                    # Dispatch Batches
                    # Stop dispatching if we have enough tasks, but finish existing futures
                    if len(frontier) + len(tasks_to_solve) < TARGET_TASKS:
                        while idle_workers_expansion and frontier:
                            worker = idle_workers_expansion.pop()
                            
                            # Create Batch
                            batch = []
                            batch_ids = []
                            while frontier and len(batch) < BATCH_SIZE:
                                node = frontier.pop(0)
                                
                                # Check terminal conditions (Depth/Budget) BEFORE sending
                                if node.budget < self.min_edge_cost or node.depth >= MAX_EXPANSION_DEPTH:
                                    node.is_terminal = True
                                    tasks_to_solve.append(node)
                                    continue
                                
                                batch.append((id(node), node.budget, node.state, node.depth))
                                batch_ids.append(id(node))
                            
                            if batch:
                                future = worker.expand_frontier_batch.remote(batch)
                                expansion_futures[future] = (worker, batch_ids)
                            else:
                                # Put worker back if frontier emptied during batch creation
                                idle_workers_expansion.append(worker)
                                break
                    
                    # If no work is happening and frontier is empty, we are done
                    if not expansion_futures:
                        break
                    
                    # Wait for results
                    ready, _ = ray.wait(list(expansion_futures.keys()), num_returns=1, timeout=1.0)
                    
                    for future in ready:
                        worker, batch_ids = expansion_futures.pop(future)
                        idle_workers_expansion.append(worker)
                        
                        try:
                            results = ray.get(future)
                            # results: list of (n_id, val, children_data, is_terminal)
                            
                            for n_id, val, children_data, is_terminal in results:
                                node = node_registry[n_id]
                                node.stopping_value = val # Cache the costly max-flow value
                                
                                if is_terminal:
                                    node.is_terminal = True
                                    if enable_memoization:
                                        memo_driver[node.key] = (val, [])
                                    tasks_to_solve.append(node)
                                else:
                                    # Create child objects
                                    for c_budget, c_state, action in children_data:
                                        child = TreeNode(
                                            c_budget, c_state, node.depth + 1, 
                                            parent=node, action_from_parent=action
                                        )
                                        node.children.append(child)
                                        node_registry[id(child)] = child
                                        frontier.append(child)
                                        
                        except Exception as e:
                            print(f"Expansion task failed: {e}")
                            # Fallback: Treat failed batch nodes as leaves to be solved later
                            for nid in batch_ids:
                                if nid in node_registry:
                                    tasks_to_solve.append(node_registry[nid])
                
                # If we stopped early due to TARGET_TASKS, move remaining frontier to solve
                tasks_to_solve.extend(frontier)
                frontier = []

            else:
                # 1. Expansion Phase (Serial)
                # We pop nodes and expand them until we have enough tasks or run out of nodes
                
                while frontier:
                    # Check for timeout
                    if time.time() - start_time > time_limit:
                        if verbose: print("Expansion phase timed out. Moving to solve...")
                        tasks_to_solve.extend(frontier)
                        frontier = []
                        break

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
                    if enable_memoization and node.key in memo_driver:
                        node.value, node.best_sequence = memo_driver[node.key]
                        node.is_terminal = True
                        # Driver pruned this whole subtree -> Report progress
                        progress_actor.increment.remote(node_volume, 0, 0, node_volume, 0)
                        continue

                    # Check base cases
                    if node.budget < self.min_edge_cost or node.depth >= 20:
                        node.is_terminal = True
                        tasks_to_solve.append(node)
                        # Sent to worker -> Worker will report progress
                        continue

                    # Apply incremental changes from root to this node (avoid large copies)
                    path_actions = []
                    curr = node
                    while curr.parent:
                        path_actions.append(curr.action_from_parent)
                        curr = curr.parent
                    path_actions.reverse()

                    # Apply increments to self.state to reach node.state
                    for a in path_actions:
                        self.state['edge_interdicted'][a] += 1
                        self.state['budget'][0] -= int(self.state['edge_costs'][a])

                    # Update flows and cache for mask_fn
                    if self.attacker_strategy == "zero_sum":
                        stop_val, self.reference_flows = self._compute_objective_and_flows()
                        stop_val = -stop_val
                    elif self.attacker_strategy == 'canalize':
                        stop_val, self.reference_flows = self._calculate_canalize_objective_and_flows()
                    elif self.attacker_strategy == 'isolate':
                        stop_val, self.reference_flows = self._calculate_isolate_objective_and_flows()
                        stop_val = -stop_val
                    elif self.attacker_strategy == 'divert':
                        stop_val, self.reference_flows = self._calculate_divert_objective_and_flows()
                    else:
                        stop_val = -float('inf')

                    node.stopping_value = stop_val

                    self._cache_flow_array()

                    action_mask = self.mask_fn()
                    valid_actions = np.where(action_mask[:self.num_both_edges] == 1)[0]

                    # Revert incremental changes to restore original driver state
                    for a in reversed(path_actions):
                        self.state['edge_interdicted'][a] -= 1
                        self.state['budget'][0] += int(self.state['edge_costs'][a])

                    if len(valid_actions) == 0:
                        node.is_terminal = True
                        if enable_memoization:
                            memo_driver[node.key] = (stop_val, [])
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
                        progress_actor.increment.remote(pruned_volume, 0, pruned_volume, 0, 0)

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
                if time.time() - start_time > time_limit:
                    if verbose: print(f"Execution phase timed out after {time.time() - start_time:.2f}s.")
                    break

                while idle_workers and pending_tasks:
                    worker = idle_workers.pop()
                    node = pending_tasks.pop(0)
                    
                    if node.value is not None:
                        idle_workers.append(worker)
                        continue
                        
                    future = worker.evaluate_subtree.remote(node.budget, node.state, node.depth, objective_tolerance)
                    running_futures[future] = (worker, node)
                
                if running_futures:
                    done_ids, _ = ray.wait(list(running_futures.keys()), num_returns=1, timeout=1.0)
                    for done_id in done_ids:
                        worker, node = running_futures.pop(done_id)
                        try:
                            val, seq = ray.get(done_id)
                            node.value = val
                            node.best_sequence = seq
                            
                            # Track best incumbent
                            prefix = []
                            curr = node
                            while curr.parent:
                                prefix.append(curr.action_from_parent)
                                curr = curr.parent
                            prefix.reverse()
                            full_seq = prefix + seq
                            
                            if val > best_incumbent_val + objective_tolerance:
                                best_incumbent_val = val
                                best_incumbent_seq = full_seq
                                alpha_actor.update.remote(val, full_seq)
                            elif abs(val - best_incumbent_val) <= objective_tolerance:
                                current_cost = sum(self.state['edge_costs'][a] for a in best_incumbent_seq)
                                new_cost = sum(self.state['edge_costs'][a] for a in full_seq)
                                if new_cost < current_cost:
                                    best_incumbent_seq = full_seq
                                    alpha_actor.update.remote(val, full_seq)

                            # Cache result in driver memo
                            if enable_memoization:
                                memo_driver[node.key] = (val, seq)
                        except Exception as e:
                            print(f"Task failed: {e}")
                            node.value = -float('inf') # Treat as failure
                        
                        idle_workers.append(worker)

            # 3. Aggregation Phase (Bottom-Up)
            # Only run fully if we haven't timed out significantly, but we can try to partial aggregate
            
            # Collect all nodes in the tree
            all_nodes = []
            q = [root_node]
            while q:
                curr = q.pop(0)
                all_nodes.append(curr)
                q.extend(curr.children)
                
            # If we didn't timeout, run formal aggregation
            if time.time() - start_time <= time_limit:
                # Sort by depth descending (deepest first)
                all_nodes.sort(key=lambda x: x.depth, reverse=True)
                
                for node in all_nodes:
                    # Compute or retrieve the value of current node (stopping value)
                    if node.stopping_value is not None:
                        val = node.stopping_value
                    else:
                        # Fallback for leaves or dynamically generated parts missing the cached value
                        if node.children or node.value is None:
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
                        else:
                            val = -float('inf') # Will not be used below since node.value is already set for leaves

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
                                
                            if child.value > best_val + objective_tolerance:
                                best_val = child.value
                                best_seq = [child.action_from_parent] + child.best_sequence
                            elif abs(child.value - best_val) <= objective_tolerance:
                                current_cost = sum(self.state['edge_costs'][a] for a in best_seq)
                                new_cost = self.state['edge_costs'][child.action_from_parent] + sum(self.state['edge_costs'][a] for a in child.best_sequence)
                                if new_cost < current_cost:
                                    best_seq = [child.action_from_parent] + child.best_sequence
                        
                        node.value = best_val
                        node.best_sequence = best_seq
                        
                        # Cache
                        if enable_memoization:
                            memo_driver[node.key] = (best_val, best_seq)
                    
                    elif node.value is None:
                        # Leaf node that wasn't solved?
                        node.value = val
                        node.best_sequence = []
                        if enable_memoization:
                            memo_driver[node.key] = (val, [])

                # Final result
                optimal_reward = root_node.value
                optimal_sequence = root_node.best_sequence
            else:
                # Timeout case: Use best incumbent found during task completion
                optimal_reward = best_incumbent_val
                optimal_sequence = best_incumbent_seq

            # Cleanup
            stop_event.set()
            poll_thread.join(timeout=2)
            final_pruned = 0
            final_invalid = 0
            final_memo = 0
            final_base = 0
            try:
                final, final_pruned, final_invalid, final_memo, final_base = ray.get(progress_actor.get_count.remote())
                pbar.update(final - last_reported)
            except Exception:
                pass
            pbar.close()

            total_states_evaluated = estimated_states
            if total_states_evaluated > 0:
                pruned_pct = (final_pruned / total_states_evaluated * 100)
                invalid_pct = (final_invalid / total_states_evaluated * 100)
                memo_pct = (final_memo / total_states_evaluated * 100)
                base_pct = (final_base / total_states_evaluated * 100)
                
                print(f"--- DP State Traversal Breakdown ---")
                print(f"Alpha Pruning States Dropped:   {final_pruned:,} ({pruned_pct:.2f}%)")
                print(f"Invalid Actions States Dropped: {final_invalid:,} ({invalid_pct:.2f}%)")
                print(f"Memoization Cache Hits:         {final_memo:,} ({memo_pct:.2f}%)")
                print(f"Leaves Actually Visited:        {final_base:,} ({base_pct:.2f}%)")
                print(f"Total States (Estimated):       {total_states_evaluated:,}")
                print(f"------------------------------------")

            if optimal_reward is not None and optimal_reward <= initial_alpha + 1.1e-4:
                print("Note: The initial alpha (from heuristic or RL) was not beaten during backward induction.")

            if self.attacker_strategy in ("zero_sum", "isolate"):
                # The DP maximizes "reward" (negative flow). 
                # We negate it here to return "positive flow" (cost) to match Gurobi.
                if optimal_reward is not None:
                    optimal_reward = -optimal_reward

            optimal_actions = [self.both_edges[idx] for idx in optimal_sequence]
            
            return optimal_reward, optimal_actions
            
        finally:
            self.state = copy.deepcopy(original_state)
            self.enable_outcome_caching = original_enable_outcome_caching
            self.reduce_flow = original_reduce_flow
            self.local_outcome_cache = copy.deepcopy(original_local_outcome_cache)
            self.outcome_memo_actors = original_outcome_memo_actors
            self.reference_flows = copy.deepcopy(original_reference_flows)
            self.reference_flows_dict = copy.deepcopy(original_reference_flows_dict)
            self.reference_start_flow = original_reference_start_flow
            self.reference_start_flows = copy.deepcopy(original_reference_start_flows)
            self.reference_obj = original_reference_obj
            self.reference_budget = original_reference_budget
            self.last_obj = original_last_obj
            self.canalize_norm_factor = original_canalize_norm_factor
            self.max_canalize_objective = original_max_canalize_objective
            self.max_divert_objective = original_max_divert_objective
            if original_cached_flow_array is not None:
                self.cached_flow_array = copy.deepcopy(original_cached_flow_array)
            elif hasattr(self, 'cached_flow_array'):
                del self.cached_flow_array

            if self.reference_flows is not None and hasattr(self, '_cache_flow_array'):
                try:
                    self._cache_flow_array()
                except Exception:
                    pass

            # SAFETY CLEANUP: Ensure all actors are killed even if exception occurs
            # This prevents zombie actors from accumulating between episodes
            for w in workers:
                try: ray.kill(w)
                except: pass
            
            if progress_actor:
                try: ray.kill(progress_actor)
                except: pass
            
            # Kill all memo actors
            for ma in memo_actors:
                try: ray.kill(ma)
                except: pass

            for actor in outcome_memo_actors:
                try: ray.kill(actor)
                except: pass
            self.outcome_memo_actors = None

            if alpha_actor:
                try: ray.kill(alpha_actor)
                except: pass