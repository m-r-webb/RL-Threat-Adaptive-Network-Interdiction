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
    def increment(self, n: int = 1):
        self.count += int(n)
    def get_count(self):
        return self.count
    def reset(self):
        self.count = 0

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
    def __init__(self, initial_alpha=-float('inf')):
        self.alpha = initial_alpha
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
                 max_depth_inner=100, outcome_memo_actor=None, outcome_memo_actors=None, alpha_actor=None,
                 enable_outcome_caching=True, enable_alpha_pruning=True):
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
                             outcome_memo_actors=None)
        
        # Set config flag
        self.env.enable_outcome_caching = enable_outcome_caching

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
        self.enable_alpha_pruning = enable_alpha_pruning
        
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

            action_mask = self.env.mask_fn()

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

            if self.enable_alpha_pruning:
                # Heuristic sorting for pruning
                # Temporarily set state for calculate_action_heuristics to see current interdictions
                self.env.state['budget'][0] = rem_budget
                self.env.state['edge_interdicted'][:] = inter_state
                
                heuristics = self.env.calculate_action_heuristics(valid_actions, current_flows, rem_budget)
                
                # Restore
                self.env.state['budget'][0] = old_budget
                self.env.state['edge_interdicted'][:] = old_interdicted
                
                # Sort descending
                sorted_indices = np.argsort(-heuristics)
                valid_actions = valid_actions[sorted_indices]
                heuristics = heuristics[sorted_indices]
            
            for i, action in enumerate(valid_actions):
                # Pruning
                if self.enable_alpha_pruning:
                     # Pruning condition using the new consolidated heuristic
                     # Added tolerace (1e-6) to tolerate floating point errors
                     if final_objective + heuristics[i] < alpha - 1e-6:
                         skipped_actions = len(valid_actions) - i
                         est_per_skipped = int(self.num_both_edges ** max(0, self.budget_levels - (d + 1)))
                         local_counter += skipped_actions * est_per_skipped
                         maybe_flush_progress()
                         #break  # ADD BACK
                         # Do not memoize results from pruned nodes as they may be valid for lower alphas
                         return best_reward, best_seq

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

# --- Mixin Class ---

class InterdictionSolverMixin:
    """
    Mixin containing optimization and machine learning solver methods for Network Interdiction.
    """

    def solve_optimal_interdiction(self, method='monolithic'):
        if self.deterministic_outcomes == True:
            # Deterministic: Solve using Model 1D
            if not hasattr(self, 'optimal_model'):
                # Initializing the model
                self.optimal_model = grb.Model("Optimal Model", env=self.GUROBI_ENV)

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
        
        else:
            # Stochastic outcomes
            # Use Ray to parallelize SAA if requested (via wrapper)
            # But here we just run sequentially or use the method arg
            # Just do standard parallel loop for evaluating candidates if needed, 
            # OR assume 'monolithic' means SAA Model 1U.
            
            # NOTE: solve_optimal_interdiction usually implies finding the best action.
            # In stochastic case, SAA is the standard method here.
            
            # Using SAA
            tasks = []
            seeds = [12345 + i for i in range(1)] # Just 1 run for now? Using internal scenarios.
            # Usually solve_stochastic_max_flow handles the SAA optimization itself.
            pass

    def _create_lp_maxflow_model(self):
        """Create a continuous Max Flow LP model for subproblems."""
        m = grb.Model("Subproblem_LP", env=self.GUROBI_ENV)
        
        # Continuous flow variables for ALL edges (forward + reverse)
        flow = m.addVars(self.all_both_edges, lb=0, name="flow")
        
        # Flow Conservation
        m.addConstrs(
            (grb.quicksum(flow[e] for e in self.edge_groups[n]['out']) == 
             grb.quicksum(flow[e] for e in self.edge_groups[n]['in'])
             for n in self.intermediate_nodes), name="conservation"
        )

        # Super Source/Sink conservation
        total_flow = m.addVar(name="total_flow")
        
        m.addConstr(grb.quicksum(flow[e] for e in self.edge_groups[1]['out']) - 
                    grb.quicksum(flow[e] for e in self.edge_groups[1]['in']) == total_flow, name="source_node")
        
        m.addConstr(grb.quicksum(flow[e] for e in self.edge_groups[self.super_sink_nodes[0]]['in']) -
                    grb.quicksum(flow[e] for e in self.edge_groups[self.super_sink_nodes[0]]['out']) == total_flow, name="sink_node")
        
        m.setObjective(total_flow, grb.GRB.MAXIMIZE)
        
        # Capacity constraints for ALL edges (will be updated)
        cap_constrs = m.addConstrs((flow[e] <= 0 for e in self.all_both_edges), name="capacity")
        
        m.update()
        return m, cap_constrs, flow

    def _solve_stochastic_decomposition(self, n_scenarios, seed, interdicted_edges):
        np.random.seed(seed)
        
        # 1. Generate Scenarios (Consistent with Monolithic)
        probs = self.state["edge_interdiction_probability"][:self.num_both_edges]
        scenario_outcomes = np.random.binomial(1, probs, size=(n_scenarios, len(self.both_edges)))
        
        # 2. Master Problem
        master = grb.Model("Master_Benders", env=self.GUROBI_ENV)
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
        sub_model, cap_constrs, flow_vars = self._create_lp_maxflow_model()
        
        # Benders Loop
        LB = -float('inf')
        UB = float('inf')
        epsilon = 1e-4
        max_iter = 100
        
        for iteration in range(max_iter):
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
                for idx, e in enumerate(self.both_edges):
                    # Cap = Original * (1 - x_hat * outcome)
                    is_blocked = (x_hat[e] > 0.5) and (scenario_outcomes[s, idx] == 1)
                    cap = 0 if is_blocked else self.edges_episode[e].capacity
                    
                    # Update Forward
                    cap_constrs[e].RHS = cap
                    
                    # Update Reverse
                    rev_e = (e[1], e[0])
                    cap_constrs[rev_e].RHS = cap
                
                sub_model.optimize()
                
                if sub_model.status == grb.GRB.OPTIMAL:
                    sub_obj = sub_model.ObjVal
                    total_flow += sub_obj
                    
                    # Retrieve primal flows to build Benders cut
                    for idx, e in enumerate(self.both_edges):
                         outcome = scenario_outcomes[s, idx]
                         
                         # Get flow on forward edge e
                         f_fwd = flow_vars[e].X
                         
                         # Get flow on reverse edge 
                         rev_e = (e[1], e[0])
                         f_rev = flow_vars[rev_e].X
                         
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

        interdicted = [e for e in self.both_edges if gamma[e].X > 0.5]
        return UB, interdicted

    def _solve_stochastic_decomposition_IM(self, n_scenarios, seed, interdicted_edges, interdicted_quantities):
        np.random.seed(seed)
        
        # 1. Generate Scenarios (Same as Monolithic IM)
        interdictable_indices = [self.edge_to_index[e] for e in self.interdictable_edges]
        p_base = self.state["edge_interdiction_probability"][interdictable_indices]
        
        k_vals = np.arange(1, self.max_interdictions + 1)
        probs = 1 - (1 - p_base[:, np.newaxis]) ** k_vals
        
        scenario_outcomes = np.random.binomial(1, probs, size=(n_scenarios, len(self.interdictable_edges), len(k_vals)))
        interdictable_edge_map = {e: i for i, e in enumerate(self.interdictable_edges)}
        
        # 2. Master Problem
        master = grb.Model("Master_Benders_IM", env=self.GUROBI_ENV)
        
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
        sub_model, cap_constrs, flow_vars = self._create_lp_maxflow_model()
        sub_model.setParam("OutputFlag", 0)
        
        # Benders Loop
        LB = -float('inf')
        UB = float('inf')
        epsilon = 1e-4
        max_iter = 100
        
        for iteration in range(max_iter):
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
                    cap_constrs[e].RHS = cap
                    cap_constrs[(e[1], e[0])].RHS = cap
                
                sub_model.optimize()
                
                if sub_model.status == grb.GRB.OPTIMAL:
                    sub_obj = sub_model.ObjVal
                    total_flow += sub_obj
                    
                    # Calculate Coefs
                    for e in self.interdictable_edges:
                        # Get flow on edge
                        f_total = flow_vars[e].X + flow_vars[(e[1], e[0])].X
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

        # Extract Solution
        interdicted = []
        quantities = []
        for e in self.interdictable_edges:
            for k in k_vals:
                if gamma[e, k].X > 0.5:
                    interdicted.append(e)
                    quantities.append(k)
                    break
                    
        return UB, interdicted, quantities

    def solve_stochastic_max_flow(self, n_scenarios = 50, seed = 173, interdicted_edges = [], method='monolithic'):
        if method == 'decomposition':
            return self._solve_stochastic_decomposition(n_scenarios, seed, interdicted_edges)

        # Optimally Solve for Stochastic Solution using Model 1U and SAA
        if not hasattr(self, 'optimal_stochastic_model'):
            # Initializing the model
            self.optimal_stochastic_model = grb.Model("Stochastic Model", env=self.GUROBI_ENV)

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
            (self.stochastic_alpha[e[1],s] - self.stochastic_alpha[e[0], s] + self.stochastic_beta[e, s] + (self.stochastic_gamma[e] * scenario_outcomes[s, edge_id]) >= 0 for s in self.scenarios for edge_id, e in enumerate(self.both_edges)), name='aabg')

        # Solving
        self.optimal_stochastic_model.optimize()

        interdicted_edges = [
            e for e in self.both_edges
            if self.stochastic_gamma[e].X > 0.5 
        ]

        return(self.optimal_stochastic_model.objVal, interdicted_edges)

    def _eval_fixed_strategy(self, n_scenarios, seed, interdicted_edges):
        """Efficiently evaluate a fixed interdiction strategy by averaging max flows."""
        np.random.seed(seed)
        
        # 1. Generate Scenarios
        probs = self.state["edge_interdiction_probability"][:self.num_both_edges]
        scenario_outcomes = np.random.binomial(1, probs, size=(n_scenarios, len(self.both_edges)))
        
        # 2. Setup Subproblem (Max Flow LP)
        sub_model, cap_constrs, _ = self._create_lp_maxflow_model()
        sub_model.setParam("OutputFlag", 0)
        
        # 3. Create blockage map
        x_fixed = {e: 0 for e in self.both_edges}
        for e in interdicted_edges:
            x_fixed[e] = 1
            
        total_flow = 0.0
        
        for s in range(n_scenarios):
             # Update Capacities
            for idx, e in enumerate(self.both_edges):
                outcome = scenario_outcomes[s, idx]
                is_blocked = (x_fixed[e] == 1) and (outcome == 1)
                
                cap = 0 if is_blocked else self.edges_episode[e].capacity
                
                cap_constrs[e].RHS = cap
                cap_constrs[(e[1], e[0])].RHS = cap
            
            sub_model.optimize()
            if sub_model.status == grb.GRB.OPTIMAL:
                total_flow += sub_model.ObjVal
                
        return total_flow / n_scenarios, interdicted_edges

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

    def train_baycik_model(self, pickle_path):
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
            for i in tqdm(range(len(states)), desc="Training RF"):
                state = states[i]
                if state is None: continue
                
                # Handle Gymnasium reset return (obs, info)
                if isinstance(state, tuple) and len(state) == 2:
                    if isinstance(state[0], dict) and 'edge_capacity' in state[0]:
                        state = state[0]

                # Load state into environment
                self.state = state
                self._cache_flow_array() # Update cache based on state
                
                # Calculate Initial Uninterdicted Flow
                # Ensure no interdictions are considered for the feature extraction
                # We temporarily clear interdictions in state dict
                temp_interdicted = self.state['edge_interdicted'].copy()
                self.state['edge_interdicted'] = np.zeros_like(temp_interdicted)
                
                # Solve max flow to get features
                _, flow_dict = self.solve_max_flow()
                
                # Restore interdictions (though usually 0 at start of episode)
                self.state['edge_interdicted'] = temp_interdicted
                
                budget = state['budget'][0]
                target_set = set(optimal_solutions[i])
                
                # Max capacity for normalization
                current_caps = state['edge_capacity'][:self.num_both_edges]
                max_net_cap = np.max(current_caps) if len(current_caps) > 0 else 1.0
                
                for idx in range(self.num_both_edges):
                    edge = self.both_edges[idx]
                    u, v = edge
                    
                    cap = state['edge_capacity'][idx]
                    cost = state['edge_costs'][idx]
                    f_val = flow_dict.get(edge, 0)
                    
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

                    # Features: Cost, Flow, Budget, Interdiction Prob + 7 New. (Removed Raw Capacity)
                    features = [cost, f_val, budget, prob_success,
                                tail_in, tail_out, head_in, head_out,
                                norm_d_src, norm_d_sink, norm_cap]

                    label = 1 if edge in target_set else 0
                    
                    X.append(features)
                    y.append(label)
                    
        finally:
            self.state = original_state
            self._cache_flow_array()
            
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        return clf

    def solve_baycik_interdiction(self, model):
        """Solve using Baycik's Random Forest Heuristic."""
        # 1. Calculate Initial Uninterdicted Flow for Features
        temp_interdicted = self.state['edge_interdicted'].copy()
        self.state['edge_interdicted'] = np.zeros_like(temp_interdicted)
        _, flow_dict = self.solve_max_flow()
        self.state['edge_interdicted'] = temp_interdicted
        
        # Static Features
        static_feats = self._compute_baycik_static_features()
        
        budget = self.state['budget'][0]
        candidates = []
        
        # Max capacity
        current_caps = self.state['edge_capacity'][:self.num_both_edges]
        max_net_cap = np.max(current_caps) if len(current_caps) > 0 else 1.0
        
        for idx in range(self.num_both_edges):
            edge = self.both_edges[idx]
            u, v = edge
            
            # Skip if already interdicted (if that's possible in usage context)
            if self.state['edge_interdicted'][idx] == 1:
                continue
                
            cap = self.state['edge_capacity'][idx]
            cost = self.state['edge_costs'][idx]
            f_val = flow_dict.get(edge, 0)
            
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

            features = [cost, f_val, budget, prob_success,
                        tail_in, tail_out, head_in, head_out,
                        norm_d_src, norm_d_sink, norm_cap]
            
            # Predict Prob of Class 1
            prob = model.predict_proba([features])[0][1]
            candidates.append({'edge': edge, 'prob': prob, 'cost': cost})
            
        # Sort by probability descending
        candidates.sort(key=lambda x: x['prob'], reverse=True)
        
        selected_edges = []
        current_spend = 0
        
        # Greedy Selection (Knapsack-like)
        for cand in candidates:
            if current_spend + cand['cost'] <= budget:
                selected_edges.append(cand['edge'])
                current_spend += cand['cost']
        
        # Evaluate
        if self.deterministic_outcomes:
            # Deterministic Evaluation
            sub, cap_cons, _ = self._create_lp_maxflow_model()
            sub.setParam("OutputFlag", 0)
            
            block_set = set(selected_edges)
            for e in self.both_edges:
                # If interdicted, 0 capacity
                cap = 0 if e in block_set else self.edges_episode[e].capacity
                cap_cons[e].RHS = cap
                cap_cons[(e[1], e[0])].RHS = cap
                
            sub.optimize()
            objective_value = sub.ObjVal
        else:
            # Stochastic Evaluation
            objective_value, _ = self._eval_fixed_strategy(n_scenarios=500, seed=123, interdicted_edges=selected_edges)
            
        return objective_value, selected_edges

    def solve_stochastic_max_flow_IM(self, n_scenarios = 50, seed = 173, interdicted_edges = [], interdicted_quantities =[], method='monolithic'):
        if method == 'decomposition':
            return self._solve_stochastic_decomposition_IM(n_scenarios, seed, interdicted_edges, interdicted_quantities)

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

    # --- Re-add solve_backward_induction_ray method to Mixin ---
    
    def solve_backward_induction_ray(self, verbose=False, n_workers=4, worker_depth=None, ray_address=None, enable_memoization=True, enable_outcome_caching=True, enable_alpha_pruning=True):
        """
        Parallelized backward induction using Ray with Adaptive Frontier Expansion.
        """
        # Import locally to ensure availability in all paths and avoid scope issues
        import copy, numpy as np, ray as _ray, time

        # init ray if not already
        if n_workers > 0 and not ray.is_initialized():
            ray.init(address=ray_address, ignore_reinit_error=True)

        # Ensure clean Gurobi model state for determinism
        self._cleanup_models()

        # Propagate caching flag to driver for heuristic usage
        self.enable_outcome_caching = enable_outcome_caching
        if self.enable_outcome_caching:
             self.local_outcome_cache = {}

        # precompute min edge cost etc
        real_edge_costs = self.state['edge_costs'][:self.num_both_edges]
        self.min_edge_cost = min(real_edge_costs[real_edge_costs > 0], default=float('inf'))
        if self.min_edge_cost == float('inf'):
            self.min_edge_cost = 1

        # Create outcome memoization actor ONLY if stochastic
        outcome_memo_actors = []
        if not self.deterministic_outcomes and enable_outcome_caching:
            num_outcome_shards = min(4, n_workers) if n_workers > 0 else 1
            outcome_memo_actors = [SharedOutcomeMemoActor.remote() for _ in range(num_outcome_shards)]
            self.outcome_memo_actors = outcome_memo_actors

        # Calculate Initial Alpha (Heuristic) - MOVED TO TOP to support Serial Mode
        initial_alpha = -float('inf')
        initial_alpha_actions = []
        if enable_alpha_pruning:
            if verbose:
                print("Running heuristic for initial alpha...")
            
            # Save state
            old_budget = self.state['budget'][0]
            old_interdicted = self.state['edge_interdicted'].copy()
            
            try:
                current_budget = old_budget
                while True:
                    # 1. Compute current state
                    if self.attacker_strategy == "zero_sum":
                        obj_val, flows = self._compute_objective_and_flows()
                        obj_val = -obj_val # Goal: Minimize flow -> Maximize -flow.
                    elif self.attacker_strategy == "isolate":
                        obj_val, flows = self._calculate_isolate_objective_and_flows()
                        obj_val = -obj_val # Goal: Minimize target flow -> Maximize -flow.
                    elif self.attacker_strategy == "canalize":
                        obj_val, flows = self._calculate_canalize_objective_and_flows()
                    elif self.attacker_strategy == "divert":
                        obj_val, flows = self._calculate_divert_objective_and_flows()
                    else:
                        obj_val, flows = -float('inf'), {}

                    # Update reference flows and cache for mask_fn to ensure valid actions are correct
                    self.reference_flows = flows
                    self._cache_flow_array()

                    # 2. Check if budget exhausted
                    if current_budget < self.min_edge_cost:
                        initial_alpha = obj_val 
                        break
                    
                    # 3. Get valid actions
                    action_mask = self.mask_fn()
                    valid_actions = np.where(action_mask[:self.num_both_edges] == 1)[0]
                    
                    if len(valid_actions) == 0:
                        initial_alpha = obj_val
                        break

                    # 4. Select best action (Heuristic: Max Flow on Edge)
                    best_action = -1
                    if valid_actions.size > 0:
                        heuristics = self.calculate_action_heuristics(valid_actions, flows, current_budget)
                        best_idx = np.argmax(heuristics)
                        best_action = valid_actions[best_idx]
                    
                    if best_action != -1:
                        # Apply action
                        self.state['edge_interdicted'][best_action] += 1
                        cost = self.state['edge_costs'][best_action]
                        self.state['budget'][0] -= cost
                        current_budget -= cost
                        initial_alpha_actions.append(self.both_edges[best_action])
                    else:
                        break
                        
            except Exception as e:
                if verbose:
                    print(f"Heuristic failed: {e}")
                initial_alpha = -float('inf')
            finally:
                if initial_alpha > -float('inf'):
                    initial_alpha -= 2.0 # Safety margin for pruning
                # Restore state
                self.state['budget'][0] = old_budget
                self.state['edge_interdicted'][:] = old_interdicted
                
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
            def dp_serial(rem_budget, inter_state, d, alpha=-float('inf')):
                key = inter_state[:self.num_both_edges].tobytes()
                
                # Volume calc for this node's potential subtree
                current_volume = int(int(self.num_both_edges) ** max(0, budget_levels - d))

                if enable_memoization and key in memo_serial:
                    pbar.update(current_volume)
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
                    pbar.update(num_invalid * child_volume)

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
                    
                    heuristics = self.calculate_action_heuristics(valid_actions, flows, rem_budget)
                    
                    # Restore state immediately
                    self.state['budget'][0] = old_budget
                    self.state['edge_interdicted'][:] = old_interdicted
                    
                    sorted_indices = np.argsort(-heuristics)
                    valid_actions = valid_actions[sorted_indices]
                    heuristics = heuristics[sorted_indices]

                for i, action in enumerate(valid_actions):
                    # Pruning
                    if enable_alpha_pruning:
                         if val + heuristics[i] < alpha- 1e-6:
                             skipped_actions = len(valid_actions) - i
                             child_volume = int(int(self.num_both_edges) ** max(0, budget_levels - (d + 1)))
                             pbar.update(skipped_actions * child_volume)
                             break

                    inter_state[action] += 1
                    new_budget = rem_budget - self.state['edge_costs'][action]
                    
                    fut_reward, fut_seq = dp_serial(new_budget, inter_state, d + 1, alpha)
                    
                    inter_state[action] -= 1
                    
                    if fut_reward > best_reward:
                        best_reward = fut_reward
                        best_seq = [action] + fut_seq
                        alpha = max(alpha, best_reward)
                        
                if enable_memoization:
                    memo_serial[key] = (best_reward, best_seq)
                return best_reward, best_seq

            # Run Serial
            t0 = time.time()
            initial_interdicted = self.state['edge_interdicted'].copy()
            initial_budget = self.state['budget'][0]
            
            opt_reward, opt_seq = dp_serial(initial_budget, initial_interdicted, 0, alpha=initial_alpha)
            
            pbar.close()

            if verbose:
                print(f"Serial execution completed in {time.time() - t0:.2f}s")
                
            optimal_actions = [self.both_edges[idx] for idx in opt_seq]
            return opt_reward, optimal_actions


        # snapshot state to send to workers
        state_snapshot = copy.deepcopy(self.state)
        seed = getattr(self, 'seed', None)

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
        
        alpha_actor = _SharedAlphaActor.remote(initial_alpha)
        
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
                max_depth_inner=100,
                outcome_memo_actors=outcome_memo_actors,
                alpha_actor=alpha_actor,
                enable_outcome_caching=enable_outcome_caching,
                enable_alpha_pruning=enable_alpha_pruning
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
                    current = ray.get(progress_actor.get_count.remote(), timeout=1)
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
            if enable_memoization and node.key in memo_driver:
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
            
            action_mask = self.mask_fn()
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
                done_ids, _ = ray.wait(list(running_futures.keys()), num_returns=1)
                for done_id in done_ids:
                    worker, node = running_futures.pop(done_id)
                    try:
                        val, seq = ray.get(done_id)
                        node.value = val
                        node.best_sequence = seq
                        
                        # Cache result in driver memo
                        if enable_memoization:
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
        self.outcome_memo_actors = None

        try: ray.kill(alpha_actor)
        except: pass

        if self.attacker_strategy in ("zero_sum", "isolate"):
            optimal_reward = -optimal_reward

        optimal_actions = [self.both_edges[idx] for idx in optimal_sequence]
        return optimal_reward, optimal_actions
