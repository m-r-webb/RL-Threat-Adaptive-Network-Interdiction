# Train an RL agent with a Pointer Network
##Inputs
graphName = "G4x5"

# Type of agent to train (uncomment only one)
#agent = "A2C"
#agent = "DQN"
agent = "MaskablePPO"
#agent = "PPO"


# Deterministic or Stochastic Outcomes?
deterministicOutcomes = False
multiple_interdiction_attempts=False
attacker_strategy = "zero_sum"  # 'canalize'  'divert'    'isolate'   'zero_sum'

if deterministicOutcomes:
    deterministicLetter = "D"
else:
    deterministicLetter = "S"

#G3x5
version = "V2" #C: Canalize, D: Divert, I: Isolate, Z: Zero-Sum 

# Model Name
model_name = f"{graphName}_{deterministicLetter}_{agent}_{attacker_strategy}_{version}"
print(model_name)
# Initial Learning Rate
initial_learning_rate = 0.0003  #0.0001

# Time Steps to Train
timesteps = 15000000

# Number of parallel cpus
n_cpus = 144  # Number of environments

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
        self.binary_embed = nn.Embedding(2, 8)
        self.max_nodes = 250  # From your max_num_nodes
        self.node_embedding = nn.Embedding(self.max_nodes + 1, 6, padding_idx=0)  # +1 for padding        
        
        # Determine input dimension based on strategy
        base_continuous = 3  # edge_capacity, edge_costs, edge_prob
        base_binary = 8  # edge_interdicted embedding
        node_features = 12  # 6(dep) + 6(arr)

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

class PointerNetwork(nn.Module):
    """
    Pointer Network implementation for edge selection with action masking
    """
    def __init__(self, input_dim, hidden_dim, num_actions, num_layers=1):  # Add num_actions parameter
        super(PointerNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions  # Store total number of actions
        self.num_layers = num_layers
        
        # Encoder LSTM (bidirectional)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  #Beta test with false someday to gauge benefit
        )
        
        # Decoder LSTM (unidirectional)
        self.decoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Attention mechanism components
        self.W1 = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)  # Encoder projection
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)      # Decoder projection
        self.v = nn.Linear(hidden_dim, 1, bias=False)                # Attention vector
        
        # Context vector for decoder initialization
        self.h_context_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.c_context_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Learnable decoder input
        self.decoder_start_input = nn.Parameter(th.randn(1, 1, input_dim) * 0.1)
        
    def forward(self, inputs, budget=None, action_masks=None, padding_mask=None):
        """
        Forward pass with packed sequences and action masking support
        """
        batch_size, seq_len, _ = inputs.shape

        # Calculate actual sequence lengths from PADDING MASK (not action masks)
        if padding_mask is not None:
            # Use padding mask to get actual sequence lengths
            seq_lengths = padding_mask.sum(dim=1).cpu().long()  # [batch]
            seq_lengths = th.clamp(seq_lengths, min=1)

        else:
            # If no mask, use full sequence length
            seq_lengths = th.full((batch_size,), seq_len, dtype=th.long, device='cpu')
    
        # Pack sequences to skip padded entries
        packed_inputs = pack_padded_sequence(
            inputs, 
            seq_lengths, 
            batch_first=True, 
            enforce_sorted=False
        )
    
        # Encode with packed sequences - LSTM only processes actual entries
        packed_encoder_outputs, (h_enc, c_enc) = self.encoder(packed_inputs)
    
        # Unpack back to padded format for attention computation
        encoder_outputs, _ = pad_packed_sequence(
            packed_encoder_outputs,
            batch_first=True,
            total_length=seq_len
        )  # [batch_size, seq_len, hidden_dim*2]
    
        # Initialize decoder state
        h_forward = h_enc[-2, :, :]
        h_backward = h_enc[-1, :, :]
        h_combined = th.cat([h_forward, h_backward], dim=1)
    
        c_forward = c_enc[-2, :, :]
        c_backward = c_enc[-1, :, :]
        c_combined = th.cat([c_forward, c_backward], dim=1)
    
        h_dec = self.h_context_proj(h_combined).unsqueeze(0)
        c_dec = self.c_context_proj(c_combined).unsqueeze(0)
    
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
        decoder_proj = self.W2(decoder_output)  # [B, 1, hidden_dim]
        energy = th.tanh(encoder_proj + decoder_proj.expand(-1, seq_len, -1))
        attention_logits = self.v(energy).squeeze(-1)  # [B, seq_len]

        # MASK PADDING POSITIONS BEFORE ADDING "DO NOTHING" ACTION
        if padding_mask is not None:
            # Ensure padding_mask matches attention_logits size
            if padding_mask.shape[1] != attention_logits.shape[1]:
                padding_mask = padding_mask[:, :attention_logits.shape[1]]
        
            # Set padded positions to very negative value
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
        
        # Apply action masking if provided
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
                 num_edges=25, edge_embedding_dim=128, hidden_dim=256,
                 net_arch=None, activation_fn=nn.ReLU, *args, **kwargs):
        
        self.num_edges = num_edges
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
            num_actions=self.action_space_size,  # Pass the actual action space size
            num_layers=1
        ).to(self.device)
        
        # Custom value network
        self.custom_value_net = nn.Sequential(
            nn.Linear(self.edge_embedding_dim + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
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
        distribution = CategoricalDistribution(self.action_space_size)  # Use action_space_size instead of num_edges
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
        distribution = CategoricalDistribution(self.action_space_size)  # Use action_space_size instead of num_edges
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

# Add action mask method to your environment or create a wrapper
def mask_fn(env):
    """
    Function to generate action mask based on your validation logic
    """
    # Get current state
    remaining_budget = env.state['budget']
    
    # Create action mask - size should match action space
    action_mask = np.ones(env.action_space.n, dtype=np.float32)
    
    # Mask interdictable edges (first 25 actions)
    for action in range(min(env.num_interdictable_edges, env.action_space.n)):
        if not env._validate_action(action, remaining_budget, env.state['edge_interdicted']):
            action_mask[action] = 0.0
    
    # The "do nothing" action (if it exists) is typically always valid
    # It should be at index env.num_interdictable_edges
    # No additional masking needed for "do nothing" action
    return action_mask
    
# Modified environment setup with action masking
def make_env():
    env = ce.CustomEnv(nodes, edges, deterministic_agent=deterministicOutcomes,
                       multiple_interdiction_attempts=multiple_interdiction_attempts,
                       attacker_strategy=attacker_strategy)
    # Wrap with ActionMasker
    env = ActionMasker(env, mask_fn)
    return env

# Calculate the correct action space size
if attacker_strategy == "zero_sum":
    action_space_size = 25  # Only interdictable edges
else:
    action_space_size = 25 + 1  # Interdictable edges + "do nothing" action

# Policy kwargs for your training setup
policy_kwargs = dict(
    features_extractor_class=PointerNetworkFeatureExtractor,
    features_extractor_kwargs={
        'edge_embedding_dim': 128,
        'hidden_dim': 256,
        'multiple_interdiction_attempts': multiple_interdiction_attempts,
        'attacker_strategy': attacker_strategy
    },
    num_edges=action_space_size,  # Use calculated size
    edge_embedding_dim=128,
    hidden_dim=256,
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
                ce.CustomEnv(nodes, edges, deterministic_agent=deterministicOutcomes,
                           multiple_interdiction_attempts=multiple_interdiction_attempts,
                           attacker_strategy=attacker_strategy),
                mask_fn
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
        batch_size=9600,  #6144
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