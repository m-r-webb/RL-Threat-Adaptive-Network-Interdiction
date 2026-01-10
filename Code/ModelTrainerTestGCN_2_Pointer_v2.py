# Train an RL agent with a Pointer Network
##Inputs
graphName = "G5x5"

# Type of agent to train (uncomment only one)
#agent = "A2C"
#agent = "DQN"
agent = "MaskablePPO"
#agent = "PPO"

version = "v01_09_GCN" #V[Month]_[Day] 

# Initial Learning Rate
initial_learning_rate = 0.0003  #0.0001

# Time Steps to Train
timesteps = 50000000

# Number of parallel cpus
n_cpus = 120  # Number of environments

env_params = {'deterministic_agent': False,
              'multiple_interdiction_attempts': False,
              'attacker_strategy': 'divert',  # canalize   isolate   divert  zero_sum
              'training_budget_range': (5, 15),  #G5x5: zero_sum/isolate: (5,15), canalize/divert: (10,20) G10x10: zero_sum/isolate: (15,30), canalize/divert: (20,40)   #UKR: zero_sum/isolate: (10,20), canalize/divert: (15,25)
              'max_path_length': 6,  #G5x5: 6,  G10x10: 13, UKR: 16
              'sample_size': None,
              'penalty_value': -0.01,
             }

if env_params['deterministic_agent']:
    deterministicLetter = "D"
else:
    deterministicLetter = "S"

if env_params['multiple_interdiction_attempts'] == True:
    MI_letter = 'M'
else:
    MI_letter = 'B'

# Model Name
model_name = f"{graphName}_{deterministicLetter}_{agent}_{env_params['attacker_strategy']}_{MI_letter}_{version}"
print(model_name)
print(env_params)

# Import all required packages
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress most logs (including CUDA errors)
import numpy as np
import pickle

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback ,BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback  # Replace EvalCallback

# Import custom environment .py file
import env_TA as ce #modified for multiple_interdictions

# Graph nodes and edges to use
node_filename = f"{graphName}_Nodes.csv"  # Dynamically include graphName
edge_filename = f"{graphName}_Edges.csv"  # Dynamically include graphName

current_dir = os.getcwd()

# Create nodes and edges
nodes, edges = ce.create_nodes_edges(node_filename, edge_filename)

#Train the model and log mean rewards during training
models_dir = os.path.join(current_dir, '..', 'Trained_RL_Models')

# Custom learning rate function
def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.torch_layers import MlpExtractor
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from typing import Dict, Any, Tuple
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class PointerNetworkFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that processes edge features for pointer network
    """
    def __init__(self, observation_space, 
                 edge_embedding_dim=128, hidden_dim=256,
                 edge_capacity_mean=50, edge_capacity_std=28.868,
                 edge_cost_mean=5, edge_cost_std=1.543,
                 budget_mean=50, budget_std=28.868,
                 multiple_interdiction_attempts=False,
                 edge_interdicted_mean=5, edge_interdicted_std=2.889,
                 attacker_strategy='zero_sum'):
        
        # Features dimension for compatibility
        super().__init__(observation_space, features_dim=hidden_dim + 1)
        
        # Register normalization parameters
        self.register_buffer('edge_capacity_mean', th.tensor(edge_capacity_mean))
        self.register_buffer('edge_capacity_std', th.tensor(edge_capacity_std))
        self.register_buffer('edge_cost_mean', th.tensor(edge_cost_mean))
        self.register_buffer('edge_cost_std', th.tensor(edge_cost_std))
        self.register_buffer('budget_mean', th.tensor(budget_mean))
        self.register_buffer('budget_std', th.tensor(budget_std))
        
        self.multiple_interdiction_attempts = multiple_interdiction_attempts
        if self.multiple_interdiction_attempts:
            self.register_buffer('edge_interdicted_mean', th.tensor(edge_interdicted_mean))
            self.register_buffer('edge_interdicted_std', th.tensor(edge_interdicted_std))
        
        self.edge_embedding_dim = edge_embedding_dim
        self.hidden_dim = hidden_dim
        self.attacker_strategy = attacker_strategy
        
        # Edge feature processors
        self.binary_embed = nn.Embedding(2, 4)
        self.max_nodes = 250  # From your max_num_nodes
        self.node_embedding = nn.Embedding(self.max_nodes + 1, 4, padding_idx=0)  # +1 for padding        
        
        # Determine input dimension based on strategy
        base_continuous = 3  # edge_capacity, edge_costs, edge_prob
        base_binary = 4 #8  # edge_interdicted embedding
        node_features = 8 #12  # 6(dep) + 6(arr)

        if attacker_strategy == 'zero_sum':
            if self.multiple_interdiction_attempts:
                input_dim = 4 + node_features  # 4(cont) + 12(nodes)
            else:
                input_dim = base_continuous + base_binary + node_features  # 3+8+12
        
        elif attacker_strategy == 'canalize':
            # Add 1 binary feature for canalize_objective
            if self.multiple_interdiction_attempts:
                input_dim = 4 + 1 + node_features  # 4(cont) + 1(canalize) + 12(nodes)
            else:
                input_dim = base_continuous + base_binary + 1 + node_features  # 3+8+1+12
        
        elif attacker_strategy == 'isolate':
            # Add 1 binary feature for isolate_objective
            if self.multiple_interdiction_attempts:
                input_dim = 4 + 1 + node_features  # 4(cont) + 1(isolate) + 12(nodes)
            else:
                input_dim = base_continuous + base_binary + 1 + node_features  # 3+8+1+12
        
        elif attacker_strategy == 'divert':
            # Add 2 binary features for divert objectives
            if self.multiple_interdiction_attempts:
                input_dim = 4 + 2 + node_features  # 4(cont) + 2(divert) + 12(nodes)
            else:
                input_dim = base_continuous + base_binary + 2 + node_features  # 3+8+2+12
        else:
            # Default (shouldn't reach here)
            input_dim = base_continuous + base_binary + node_features
            
        # Edge embedding layer
        self.edge_embedding = nn.Sequential(
            nn.Linear(input_dim, edge_embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(edge_embedding_dim)
        )
        
        # Store processed features for pointer network
        self._last_edge_embeddings = None
        self._last_budget = None
        self._last_sequence_length = None
        
    def forward(self, observations):
        # Process edge features
        edge_capacity = th.as_tensor(observations['edge_capacity'], dtype=th.float32)
        edge_capacity = (edge_capacity - self.edge_capacity_mean) / (self.edge_capacity_std + 1e-8)
    
        edge_costs = th.as_tensor(observations['edge_costs'], dtype=th.float32)
        edge_costs = (edge_costs - self.edge_cost_mean) / (self.edge_cost_std + 1e-8)
    
        edge_prob = th.as_tensor(observations['edge_interdiction_probability'], dtype=th.float32)

        # Extract padding mask
        padding_mask = th.as_tensor(observations['padding_mask'], dtype=th.float32)
        
        if self.multiple_interdiction_attempts:
            edge_interdicted = th.as_tensor(observations['edge_interdicted'], dtype=th.float32)
            edge_interdicted = (edge_interdicted - self.edge_interdicted_mean) / (self.edge_interdicted_std + 1e-8)
        else:
            edge_interdicted = th.as_tensor(observations['edge_interdicted'], dtype=th.long)
    
        budget = th.as_tensor(observations['budget'], dtype=th.float32)
        budget = (budget - self.budget_mean) / (self.budget_std + 1e-8)
    
        dep_nodes = th.as_tensor(observations['edge_departure_node'], dtype=th.long)
        arr_nodes = th.as_tensor(observations['edge_arrival_node'], dtype=th.long)
    
        # Create node embeddings
        dep_emb = self.node_embedding(dep_nodes)  
        arr_emb = self.node_embedding(arr_nodes)
    
        # Extract strategy-specific objectives
        objective_features = []
        if self.attacker_strategy == 'canalize':
            # Make sure this key exists in observations
            canalize_obj = th.as_tensor(observations['canalize_objective'], dtype=th.float32).unsqueeze(-1)
            objective_features.append(canalize_obj)
        elif self.attacker_strategy == 'isolate':
            isolate_obj = th.as_tensor(observations['isolate_objective'], dtype=th.float32).unsqueeze(-1)
            objective_features.append(isolate_obj)
        elif self.attacker_strategy == 'divert':
            divert_from = th.as_tensor(observations['divert_from_objective'], dtype=th.float32).unsqueeze(-1)
            divert_to = th.as_tensor(observations['divert_to_objective'], dtype=th.float32).unsqueeze(-1)
            objective_features.extend([divert_from, divert_to])
    
        # Combine features
        if self.multiple_interdiction_attempts:
            cont_features = th.stack([edge_capacity, edge_costs, edge_prob, edge_interdicted], dim=-1)
            if objective_features:
                combined = th.cat([cont_features] + objective_features + [dep_emb, arr_emb], dim=-1)
            else:
                combined = th.cat([cont_features, dep_emb, arr_emb], dim=-1)
        else:
            cont_features = th.stack([edge_capacity, edge_costs, edge_prob], dim=-1)
            binary_emb = self.binary_embed(edge_interdicted)
            if objective_features:
                combined = th.cat([cont_features, binary_emb] + objective_features + [dep_emb, arr_emb], dim=-1)
            else:
                combined = th.cat([cont_features, binary_emb, dep_emb, arr_emb], dim=-1)
    
        # Generate edge embeddings
        edge_embeddings = self.edge_embedding(combined)
    
        # Store for pointer network
        self._last_edge_embeddings = edge_embeddings
        self._last_budget = budget
        self._last_padding_mask = padding_mask  # store the mask
        self._last_sequence_length = edge_embeddings.shape[1]
    
        # Return features for SB3
        batch_size = edge_embeddings.shape[0]
        budget_reshaped = budget.reshape(batch_size, -1)
        return th.cat([edge_embeddings.mean(dim=1), budget_reshaped], dim=-1)


class GraphConvLayer(nn.Module):
    """Graph Convolutional Layer with message passing"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.self_linear = nn.Linear(in_features, out_features)  # Separate transform for self
        
    def forward(self, node_features, adj_matrix):
        """
        node_features: [batch, num_nodes, in_features]
        adj_matrix: [batch, num_nodes, num_nodes] - normalized adjacency
        """
        # Message passing: aggregate neighbor features
        aggregated = th.bmm(adj_matrix, node_features)  # [batch, num_nodes, in_features]
        # Transform neighbor aggregation
        neighbor_out = self.linear(aggregated)
        # Transform self features
        self_out = self.self_linear(node_features)
        # Combine and activate
        return F.relu(neighbor_out + self_out)


class GCNFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using Graph Convolutional Network to process graph structure.
    GCN learns node representations by aggregating neighbor information, then
    combines these with edge features for the pointer network.
    """
    def __init__(self, observation_space, 
                 edge_embedding_dim=128, hidden_dim=256,
                 gcn_hidden_dim=64, num_gcn_layers=2,
                 max_nodes=500,  # Increased to accommodate larger node indices
                 edge_capacity_mean=50, edge_capacity_std=28.868,
                 edge_cost_mean=5, edge_cost_std=1.543,
                 budget_mean=50, budget_std=28.868,
                 multiple_interdiction_attempts=False,
                 edge_interdicted_mean=5, edge_interdicted_std=2.889,
                 attacker_strategy='zero_sum'):
        
        # Features dimension for compatibility
        super().__init__(observation_space, features_dim=hidden_dim + 1)
        
        # Register normalization parameters as buffers (move to correct device)
        self.register_buffer('edge_capacity_mean', th.tensor(edge_capacity_mean))
        self.register_buffer('edge_capacity_std', th.tensor(edge_capacity_std))
        self.register_buffer('edge_cost_mean', th.tensor(edge_cost_mean))
        self.register_buffer('edge_cost_std', th.tensor(edge_cost_std))
        self.register_buffer('budget_mean', th.tensor(budget_mean))
        self.register_buffer('budget_std', th.tensor(budget_std))
        
        self.multiple_interdiction_attempts = multiple_interdiction_attempts
        if self.multiple_interdiction_attempts:
            self.register_buffer('edge_interdicted_mean', th.tensor(edge_interdicted_mean))
            self.register_buffer('edge_interdicted_std', th.tensor(edge_interdicted_std))
        
        self.max_nodes = max_nodes
        self.gcn_hidden_dim = gcn_hidden_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.hidden_dim = hidden_dim
        self.attacker_strategy = attacker_strategy
        
        # Initial node feature dimension
        initial_node_dim = 16
        
        # Node feature initialization (learnable embedding)
        self.node_init_embedding = nn.Embedding(max_nodes + 1, initial_node_dim, padding_idx=0)
        
        # GCN layers for node representation learning
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GraphConvLayer(initial_node_dim, gcn_hidden_dim))
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GraphConvLayer(gcn_hidden_dim, gcn_hidden_dim))
        
        # Layer norm after GCN
        self.gcn_layer_norm = nn.LayerNorm(gcn_hidden_dim)
        
        # Binary embedding for edge_interdicted (when not multiple attempts)
        self.binary_embed = nn.Embedding(2, 4)
        
        # Calculate edge input dimension
        # Continuous features: capacity, cost, probability, (+ interdicted if multiple attempts)
        num_continuous = 4 if multiple_interdiction_attempts else 3
        num_binary_embed = 0 if multiple_interdiction_attempts else 4
        
        # Strategy-specific objective features
        if attacker_strategy == 'divert':
            num_objective_features = 2  # divert_from, divert_to
        elif attacker_strategy in ['canalize', 'isolate']:
            num_objective_features = 1
        else:
            num_objective_features = 0
        
        # Edge input: continuous + binary embed + objective features + GCN node features (src + dst)
        edge_input_dim = (num_continuous + num_binary_embed + 
                          num_objective_features + 2 * gcn_hidden_dim)
        
        # Edge embedding network
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_input_dim, edge_embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(edge_embedding_dim)
        )
        
        # Storage for pointer network
        self._last_edge_embeddings = None
        self._last_budget = None
        self._last_padding_mask = None
        self._last_sequence_length = None
        
    def _build_adjacency_matrix(self, dep_nodes, arr_nodes, padding_mask, batch_size, device):
        """
        Build normalized adjacency matrix from edge list.
        Uses symmetric normalization: D^(-1/2) * A * D^(-1/2)
        """
        # Dynamically determine actual max node index needed
        max_node_in_batch = max(dep_nodes.max().item(), arr_nodes.max().item()) + 1
        actual_max_nodes = max(self.max_nodes, max_node_in_batch)
        
        adj = th.zeros(batch_size, actual_max_nodes, actual_max_nodes, device=device)
        
        # Create edge indices for batch indexing
        batch_indices = th.arange(batch_size, device=device).unsqueeze(1).expand(-1, dep_nodes.shape[1])
        valid_mask = padding_mask.bool()
        
        # Flatten for indexing
        b_idx = batch_indices[valid_mask]
        src_idx = dep_nodes[valid_mask]
        dst_idx = arr_nodes[valid_mask]
        
        # Add edges (undirected for GCN message passing)
        adj[b_idx, src_idx, dst_idx] = 1.0
        adj[b_idx, dst_idx, src_idx] = 1.0
        
        # Add self-loops
        self_loop = th.eye(actual_max_nodes, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        adj = adj + self_loop
        
        # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        degree_inv_sqrt = degree.pow(-0.5)
        adj = adj * degree_inv_sqrt
        adj = adj * degree_inv_sqrt.transpose(-1, -2)
        
        # Store actual size for node features
        self._actual_max_nodes = actual_max_nodes
        
        return adj
        
    def forward(self, observations):
        device = next(self.parameters()).device
        
        # Extract and move observations to correct device
        edge_capacity = th.as_tensor(observations['edge_capacity'], dtype=th.float32, device=device)
        edge_capacity = (edge_capacity - self.edge_capacity_mean) / (self.edge_capacity_std + 1e-8)
        
        edge_costs = th.as_tensor(observations['edge_costs'], dtype=th.float32, device=device)
        edge_costs = (edge_costs - self.edge_cost_mean) / (self.edge_cost_std + 1e-8)
        
        edge_prob = th.as_tensor(observations['edge_interdiction_probability'], dtype=th.float32, device=device)
        
        padding_mask = th.as_tensor(observations['padding_mask'], dtype=th.float32, device=device)
        
        if self.multiple_interdiction_attempts:
            edge_interdicted = th.as_tensor(observations['edge_interdicted'], dtype=th.float32, device=device)
            edge_interdicted = (edge_interdicted - self.edge_interdicted_mean) / (self.edge_interdicted_std + 1e-8)
        else:
            edge_interdicted = th.as_tensor(observations['edge_interdicted'], dtype=th.long, device=device)
        
        budget = th.as_tensor(observations['budget'], dtype=th.float32, device=device)
        budget = (budget - self.budget_mean) / (self.budget_std + 1e-8)
        
        dep_nodes = th.as_tensor(observations['edge_departure_node'], dtype=th.long, device=device)
        arr_nodes = th.as_tensor(observations['edge_arrival_node'], dtype=th.long, device=device)
        
        batch_size = edge_capacity.shape[0]
        num_edges = dep_nodes.shape[1]
        
        # ===== GCN Processing =====
        # Build adjacency matrix from edge structure
        adj_matrix = self._build_adjacency_matrix(dep_nodes, arr_nodes, padding_mask, batch_size, device)
        
        # Use actual max nodes determined from batch
        actual_max_nodes = self._actual_max_nodes
        
        # Initialize node features with learnable embeddings
        # Clamp indices to embedding size and handle overflow with modulo
        node_indices = th.arange(actual_max_nodes, device=device).unsqueeze(0).expand(batch_size, -1)
        node_indices_clamped = node_indices % (self.max_nodes + 1)  # Wrap around for embedding lookup
        node_features = self.node_init_embedding(node_indices_clamped)  # [batch, actual_max_nodes, initial_dim]
        
        # Apply GCN layers with residual connections
        for i, gcn_layer in enumerate(self.gcn_layers):
            new_features = gcn_layer(node_features, adj_matrix)
            if i > 0 and node_features.shape[-1] == new_features.shape[-1]:
                node_features = node_features + new_features  # Residual connection
            else:
                node_features = new_features
        
        # Apply layer norm
        node_features = self.gcn_layer_norm(node_features)
        
        # ===== Gather GCN node features for each edge's endpoints =====
        # dep_nodes/arr_nodes: [batch, num_edges]
        dep_features = th.gather(
            node_features, 1, 
            dep_nodes.unsqueeze(-1).expand(-1, -1, self.gcn_hidden_dim)
        )  # [batch, num_edges, gcn_hidden_dim]
        arr_features = th.gather(
            node_features, 1,
            arr_nodes.unsqueeze(-1).expand(-1, -1, self.gcn_hidden_dim)
        )  # [batch, num_edges, gcn_hidden_dim]
        
        # ===== Combine with edge features =====
        # Extract strategy-specific objectives
        objective_features = []
        if self.attacker_strategy == 'canalize':
            canalize_obj = th.as_tensor(observations['canalize_objective'], dtype=th.float32, device=device).unsqueeze(-1)
            objective_features.append(canalize_obj)
        elif self.attacker_strategy == 'isolate':
            isolate_obj = th.as_tensor(observations['isolate_objective'], dtype=th.float32, device=device).unsqueeze(-1)
            objective_features.append(isolate_obj)
        elif self.attacker_strategy == 'divert':
            divert_from = th.as_tensor(observations['divert_from_objective'], dtype=th.float32, device=device).unsqueeze(-1)
            divert_to = th.as_tensor(observations['divert_to_objective'], dtype=th.float32, device=device).unsqueeze(-1)
            objective_features.extend([divert_from, divert_to])
        
        # Combine continuous edge features
        if self.multiple_interdiction_attempts:
            cont_features = th.stack([edge_capacity, edge_costs, edge_prob, edge_interdicted], dim=-1)
            if objective_features:
                combined = th.cat([cont_features] + objective_features + [dep_features, arr_features], dim=-1)
            else:
                combined = th.cat([cont_features, dep_features, arr_features], dim=-1)
        else:
            cont_features = th.stack([edge_capacity, edge_costs, edge_prob], dim=-1)
            binary_emb = self.binary_embed(edge_interdicted)
            if objective_features:
                combined = th.cat([cont_features, binary_emb] + objective_features + [dep_features, arr_features], dim=-1)
            else:
                combined = th.cat([cont_features, binary_emb, dep_features, arr_features], dim=-1)
        
        # Generate edge embeddings
        edge_embeddings = self.edge_embedding(combined)
        
        # Store for pointer network
        self._last_edge_embeddings = edge_embeddings
        self._last_budget = budget
        self._last_padding_mask = padding_mask
        self._last_sequence_length = edge_embeddings.shape[1]
        
        # Return features for SB3
        budget_reshaped = budget.reshape(batch_size, -1)
        return th.cat([edge_embeddings.mean(dim=1), budget_reshaped], dim=-1)


class PointerNetwork(nn.Module):
    """
    Pointer Network implementation for edge selection with action masking
    """
    def __init__(self, input_dim, hidden_dim, num_actions, num_layers=1, bidirectional=True):
        super(PointerNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Decoder LSTM (always unidirectional)
        self.decoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Attention mechanism components
        # FIXED: Account for bidirectional encoder output
        encoder_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.W1 = nn.Linear(encoder_output_dim, hidden_dim, bias=False)  # Encoder projection
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)           # Decoder projection
        self.v = nn.Linear(hidden_dim, 1, bias=False)                     # Attention vector
        
        # Context vector for decoder initialization
        # FIXED: Input size depends on bidirectionality
        context_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.h_context_proj = nn.Linear(context_input_dim, hidden_dim)
        self.c_context_proj = nn.Linear(context_input_dim, hidden_dim)
        
        # Learnable decoder input
        self.decoder_start_input = nn.Parameter(th.randn(1, 1, input_dim) * 0.1)
        
    def forward(self, inputs, budget=None, action_masks=None, padding_mask=None):
        """
        Forward pass with packed sequences and action masking support
        """
        batch_size, seq_len, _ = inputs.shape

        # Calculate actual sequence lengths from PADDING MASK
        if padding_mask is not None:
            seq_lengths = padding_mask.sum(dim=1).cpu().long()
            seq_lengths = th.clamp(seq_lengths, min=1)
        else:
            seq_lengths = th.full((batch_size,), seq_len, dtype=th.long, device='cpu')
    
        # Pack sequences to skip padded entries
        packed_inputs = pack_padded_sequence(
            inputs, 
            seq_lengths, 
            batch_first=True, 
            enforce_sorted=False
        )
    
        # Encode with packed sequences
        packed_encoder_outputs, (h_enc, c_enc) = self.encoder(packed_inputs)
    
        # Unpack back to padded format for attention computation
        encoder_outputs, _ = pad_packed_sequence(
            packed_encoder_outputs,
            batch_first=True,
            total_length=seq_len
        )  # [batch_size, seq_len, hidden_dim*2 if bidirectional else hidden_dim]
    
        # Initialize decoder state
        if self.bidirectional:
            # Combine forward and backward hidden states
            h_forward = h_enc[-2, :, :]
            h_backward = h_enc[-1, :, :]
            h_combined = th.cat([h_forward, h_backward], dim=1)
        
            c_forward = c_enc[-2, :, :]
            c_backward = c_enc[-1, :, :]
            c_combined = th.cat([c_forward, c_backward], dim=1)
        
            h_dec = self.h_context_proj(h_combined).unsqueeze(0)
            c_dec = self.c_context_proj(c_combined).unsqueeze(0)
        else:
            # Unidirectional: just use the last layer
            h_dec = self.h_context_proj(h_enc[-1, :, :]).unsqueeze(0)
            c_dec = self.c_context_proj(c_enc[-1, :, :]).unsqueeze(0)
    
        # Create decoder input
        decoder_input = self.decoder_start_input.expand(batch_size, -1, -1)
    
        # Add budget information if available
        if budget is not None:
            budget_expanded = budget.view(batch_size, 1, 1).expand(batch_size, 1, self.input_dim)
            decoder_input = decoder_input + 0.1 * budget_expanded
    
        # Single decoder step
        decoder_output, _ = self.decoder(decoder_input, (h_dec, c_dec))
    
        # Compute attention scores
        encoder_proj = self.W1(encoder_outputs)  # [B, seq_len, hidden_dim]
        decoder_proj = self.W2(decoder_output)    # [B, 1, hidden_dim]
        energy = th.tanh(encoder_proj + decoder_proj.expand(-1, seq_len, -1))
        attention_logits = self.v(energy).squeeze(-1)  # [B, seq_len]

        # MASK PADDING POSITIONS
        if padding_mask is not None:
            if padding_mask.shape[1] != attention_logits.shape[1]:
                padding_mask = padding_mask[:, :attention_logits.shape[1]]
        
            mask_value = th.finfo(attention_logits.dtype).min
            attention_logits = th.where(
                padding_mask.bool(), 
                attention_logits, 
                mask_value
            )
    
        # Add "do nothing" action if needed
        if self.num_actions > seq_len:
            extra_actions = self.num_actions - seq_len
            extra_logits = th.zeros(batch_size, extra_actions, 
                                   device=attention_logits.device, 
                                   dtype=attention_logits.dtype)
            attention_logits = th.cat([attention_logits, extra_logits], dim=1)
        
        # Apply action masking
        if action_masks is not None:
            if isinstance(action_masks, np.ndarray):
                action_masks = th.from_numpy(action_masks).to(
                    device=attention_logits.device, dtype=attention_logits.dtype)
            elif not isinstance(action_masks, th.Tensor):
                action_masks = th.tensor(action_masks, 
                                    device=attention_logits.device, 
                                    dtype=attention_logits.dtype)
            else:
                action_masks = action_masks.to(device=attention_logits.device, dtype=attention_logits.dtype)
        
            if len(action_masks.shape) == 1:
                action_masks = action_masks.unsqueeze(0).expand(batch_size, -1)
        
            if attention_logits.shape[1] != action_masks.shape[1]:
                raise ValueError(f"Attention logits size {attention_logits.shape[1]} "
                               f"doesn't match action mask size {action_masks.shape[1]}")
        
            mask_value = th.finfo(attention_logits.dtype).min
            attention_logits = th.where(action_masks > 0, attention_logits, mask_value)
    
        return attention_logits
        
class MaskablePointerNetworkPolicy(MaskableActorCriticPolicy):
    """
    Maskable policy that uses Pointer Network for edge selection
    """
    def __init__(self, observation_space, action_space, lr_schedule,
                 edge_embedding_dim=128, hidden_dim=256, 
                 net_arch=None, activation_fn=nn.ReLU, *args, **kwargs):
        
        self.edge_embedding_dim = edge_embedding_dim
        self.hidden_dim = hidden_dim
        self.action_space_size = action_space.n  # Get actual action space size

        
        super().__init__(observation_space, action_space, lr_schedule,
                        net_arch=net_arch, activation_fn=activation_fn,
                        *args, **kwargs)
    
    def _build_mlp_extractor(self) -> None:
        """
        Build MLP extractor and pointer network
        """
        # Minimal MLP extractor for SB3 compatibility
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=[64, 64],
            activation_fn=self.activation_fn,
            device=self.device,
        )

        # Pointer Network for action selection - use actual action space size
        self.pointer_network = PointerNetwork(
            input_dim=self.edge_embedding_dim,
            hidden_dim=self.hidden_dim,
            num_actions=self.action_space_size,
            num_layers=1,
            bidirectional=True  # Explicitly set
        ).to(self.device)
        
        # Custom value network
        self.custom_value_net = nn.Sequential(
            nn.Linear(self.edge_embedding_dim + 1, 128),
            nn.ReLU(),
#            nn.Linear(256, 128),
#            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)
    
    def forward(self, obs, deterministic: bool = False, action_masks=None):
        """Forward pass using pointer network with action masking"""
        # First, extract features to populate embeddings
        features = self.extract_features(obs)
    
        # Then check if extraction was successful BEFORE trying to use them
        if self.features_extractor._last_edge_embeddings is None:
            raise ValueError("Edge embeddings not found in feature extractor")
    
        # Now safely retrieve the embeddings
        edge_embeddings = self.features_extractor._last_edge_embeddings
        budget = self.features_extractor._last_budget
        padding_mask = self.features_extractor._last_padding_mask  # retrieve mask
    
        attention_logits = self.pointer_network(edge_embeddings, budget, action_masks, padding_mask)
        distribution = CategoricalDistribution(self.action_space_size)  
        distribution = distribution.proba_distribution(action_logits=attention_logits)
    
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
    
        global_features = th.cat([edge_embeddings.mean(dim=1), budget.reshape(-1, 1)], dim=-1)
        values = self.custom_value_net(global_features)
    
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions, action_masks=None):
        """Evaluate actions for training with action masking"""
        features = self.extract_features(obs)
        edge_embeddings = self.features_extractor._last_edge_embeddings
        budget = self.features_extractor._last_budget
        padding_mask = self.features_extractor._last_padding_mask

        attention_logits = self.pointer_network(edge_embeddings, budget, action_masks, padding_mask)
        distribution = CategoricalDistribution(self.action_space_size)  
        distribution = distribution.proba_distribution(action_logits=attention_logits)
    
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
    
        global_features = th.cat([edge_embeddings.mean(dim=1), budget.reshape(-1, 1)], dim=-1)
        values = self.custom_value_net(global_features)
    
        return values, log_prob, entropy
    
    def get_distribution(self, obs, action_masks=None):
        """
        OVERRIDE: Get action distribution using pointer network
        This method is called during evaluation and prediction
        """
        features = self.extract_features(obs)
        edge_embeddings = self.features_extractor._last_edge_embeddings
        budget = self.features_extractor._last_budget
        padding_mask = self.features_extractor._last_padding_mask
        
        if edge_embeddings is None:
            raise ValueError("Edge embeddings not found in feature extractor")
        
        attention_logits = self.pointer_network(edge_embeddings, budget, action_masks, padding_mask)
        distribution = CategoricalDistribution(self.action_space_size)  # Use actual action space size
        return distribution.proba_distribution(action_logits=attention_logits)
    
    def predict_values(self, obs):
        """OVERRIDE: Predict values using our custom value network"""
        features = self.extract_features(obs)
        edge_embeddings = self.features_extractor._last_edge_embeddings
        budget = self.features_extractor._last_budget
        
        if edge_embeddings is None:
            return super().predict_values(obs)
        
        global_features = th.cat([edge_embeddings.mean(dim=1), budget.reshape(-1, 1)], dim=-1)
        return self.custom_value_net(global_features)

# Modified environment setup with action masking
def make_env():
    env = ce.CustomEnv(nodes, edges, **env_params)
    # Wrap with ActionMasker
    env = ActionMasker(env, lambda env: env.mask_fn())
    return env

# Policy kwargs for your training setup
# Switch between PointerNetworkFeatureExtractor and GCNFeatureExtractor:
# - PointerNetworkFeatureExtractor: Uses simple node embeddings
# - GCNFeatureExtractor: Uses Graph Convolutional Network to learn node representations
USE_GCN = True  # Set to False to use original PointerNetworkFeatureExtractor

if USE_GCN:
    policy_kwargs = dict(
        features_extractor_class=GCNFeatureExtractor,
        features_extractor_kwargs={
            'edge_embedding_dim': 64,
            'hidden_dim': 128,
            'gcn_hidden_dim': 32,       # Hidden dimension for GCN layers
            'num_gcn_layers': 2,         # Number of GCN message passing layers
            'max_nodes': 500,            # Maximum number of nodes in graph (increased for safety)
            'multiple_interdiction_attempts': env_params['multiple_interdiction_attempts'],
            'attacker_strategy': env_params['attacker_strategy']
        },
        edge_embedding_dim=64,
        hidden_dim=128,
        net_arch=[128, 128],
        activation_fn=nn.ReLU,
    )
else:
    policy_kwargs = dict(
        features_extractor_class=PointerNetworkFeatureExtractor,
        features_extractor_kwargs={
            'edge_embedding_dim': 64,
            'hidden_dim': 128,
            'multiple_interdiction_attempts': env_params['multiple_interdiction_attempts'],
            'attacker_strategy': env_params['attacker_strategy']
        },
        edge_embedding_dim=64,
        hidden_dim=128,
        net_arch=[128, 128],
        activation_fn=nn.ReLU,
    )

# Update your training setup to use MaskablePPO
if __name__ == "__main__":
    num_envs = n_cpus
    envs = [make_env for _ in range(num_envs)]
    vec_env = SubprocVecEnv(envs)
    
    # Create evaluation environment with action masking
    eval_env = DummyVecEnv([
        lambda: Monitor(
            ActionMasker(
                ce.CustomEnv(nodes, edges, **env_params),
                lambda env: env.mask_fn()
            )
        )
    ])
    
    # Use MaskableEvalCallback instead of regular EvalCallback
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=f"{models_dir}/{model_name}",
        log_path=f"{models_dir}/{model_name}",
        eval_freq=700,
        n_eval_episodes=576,
        deterministic=True,
        render=False,
        verbose=False
    )
    
    # Use MaskablePPO instead of regular PPO
    model = MaskablePPO(
        policy=MaskablePointerNetworkPolicy,
        env=vec_env,
        verbose=0,
        learning_rate=linear_schedule(initial_learning_rate),
        n_steps=200,  #128
        n_epochs=10,   #5
        ent_coef=0.01,  #0.05
        batch_size=2400,  #6144
        gamma=0.99,
        policy_kwargs=policy_kwargs
    )
    
    # Train with callbacks
    checkpoint_callback = CheckpointCallback(save_freq=1750, 
                                           save_path=f"{models_dir}/{model_name}",
                                           name_prefix=model_name)
    
    model.learn(total_timesteps=timesteps, 
               callback=[checkpoint_callback, eval_callback],
               progress_bar=True)