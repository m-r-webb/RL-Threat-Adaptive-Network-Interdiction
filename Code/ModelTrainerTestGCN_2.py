#Simple Features Extractor PPO_EX003GCN - Edge to 128 embedding
#LSTM Features Extractor PPO_EX003LSTM - Edge to 128 embedding with budget in LSTM
#LSTMv2 Features Extractor - PPO_EX003LSTMv2 - Edge to 128 embedding in LSTM with budget cell state

##Inputs
graphName = "G3x4"

# Type of agent to train (uncomment only one)
#agent = "A2C"
agent = "DQN"
#agent = "PPO"
#agent = "MaskablePPO"


# Deterministic or Stochastic Outcomes?
deterministicOutcomes = False
fixedCosts = False
multiple_interdiction_attempts=False
min_training_budget=3
max_training_budget=15

if deterministicOutcomes:
    deterministicLetter = "D"
else:
    deterministicLetter = "S"

#G5x5
#version = "EX001" #Stochastic PPO, DQN
version = "EX001A1" #Deterministic PPO Unlettered: 5x5, B: 50x50, C:normalized budget DQN A: 512,512,265 network B: 512,512,265,256 network, increased exploration to .9

# Model Name
model_name = f"{graphName}_{deterministicLetter}_{agent}_{version}"
print(model_name)
# Initial Learning Rate
initial_learning_rate = 0.0001  #B: 0.0003

# Time Steps to Train
timesteps = 30000000

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

import torch as th
import torch.nn as nn

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback ,BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback  # Replace EvalCallback

# Import custom environment .py file
import env_IM as ce #modified for multiple_interdictions

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

def make_env():
    env = ce.CustomEnv(nodes, edges, deterministic_agent=deterministicOutcomes, fixed_costs=fixedCosts, curriculum_training=False, min_training_budget=min_training_budget,
                       max_training_budget = max_training_budget, multiple_interdiction_attempts=multiple_interdiction_attempts)
    return env
    
#IMPLEMENT CURRICULUM LEARNING
class CurriculumCallback(BaseCallback):
    def __init__(self, decay_freq=576000, verbose=0):
        super().__init__(verbose)
        self.decay_freq = decay_freq
        
    def _on_step(self) -> bool:
        if self.num_timesteps % self.decay_freq == 0:
            # Update all environments simultaneously
            self.training_env.env_method("decay_zero_prob")
        if self.num_timesteps % (self.decay_freq*2) == 0:
            self.training_env.env_method("increase_budget")
            # Get current probability for monitoring
            #zero_probs = self.training_env.get_attr("zero_prob")
            #self.logger.record("curriculum/zero_prob", np.mean(zero_probs))
        return True
#END IMPLEMENT CURRICULUM LEARNING

#Custom Features Extractor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from torch.nn import Linear
import torch.nn.functional as F

class SimpleFeatureExtractor(BaseFeaturesExtractor):    #IM Update
    def __init__(self, observation_space, num_edges=17, num_nodes=14, embedding_dim=128,
                 edge_capacity_mean=50, edge_capacity_std=28.868,
                 edge_cost_mean=4, edge_cost_std= 0.577,
                 budget_mean = 9, budget_std= 3.46,  #20, budget_std= 11.547, #
                 multiple_interdiction_attempts = True,
                 edge_interdicted_mean = 5, edge_interdicted_std= 2.889,
                ):
        super().__init__(observation_space, features_dim=num_edges * embedding_dim + 1)

        #Register normalization parameters as buffers
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
        
        self.num_edges = num_edges
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes
        
        # Feature processors
        self.binary_embed = nn.Embedding(2, 8)  # For edge_interdicted
        self.node_embed = nn.Embedding(self.num_nodes, 16)  # For dep/arr nodes

        if self.multiple_interdiction_attempts:
            self.edge_proj = nn.Sequential(
                nn.Linear(4 + 16 + 16, embedding_dim),  # 4(cont) + 16(dep) + 16(arr)
                nn.LeakyReLU(negative_slope=0.01),
                nn.LayerNorm(embedding_dim)
            )
        else:
            self.edge_proj = nn.Sequential(
                nn.Linear(3 + 8 + 16 + 16, embedding_dim),  # 3(cont) + 8(binary) + 16(dep) + 16(arr)
                nn.LeakyReLU(negative_slope=0.01),
                nn.LayerNorm(embedding_dim)
            )

    def forward(self, observations):
        ## Original feature processing
        edge_capacity = th.as_tensor(observations['edge_capacity'], dtype=th.float32)
        edge_capacity = (edge_capacity - self.edge_capacity_mean) / (self.edge_capacity_std + 1e-8)

        edge_costs = th.as_tensor(observations['edge_costs'], dtype=th.float32)
        edge_costs = (edge_costs - self.edge_cost_mean) / (self.edge_cost_std + 1e-8)
        
        edge_prob = th.as_tensor(observations['edge_interdiction_probability'], dtype=th.float32)

        if self.multiple_interdiction_attempts:
            edge_interdicted = th.as_tensor(observations['edge_interdicted'], dtype=th.float32)
            edge_interdicted = (edge_interdicted - self.edge_interdicted_mean) / (self.edge_interdicted_std + 1e-8)
        else:
            edge_interdicted = th.as_tensor(observations['edge_interdicted'], dtype=th.long)
        
        budget = th.as_tensor(observations['budget'], dtype=th.float32)
        budget = (budget - self.budget_mean) / (self.budget_std + 1e-8)
        
        dep_nodes = th.as_tensor(observations['edge_departure_node'], dtype=th.long)
        arr_nodes = th.as_tensor(observations['edge_arrival_node'], dtype=th.long)
        
        # Process features, and combine
        dep_emb = self.node_embed(dep_nodes)  # [B, edges, 16]
        arr_emb = self.node_embed(arr_nodes)  # [B, edges, 16]

        
        if self.multiple_interdiction_attempts:
            cont_features = th.stack([edge_capacity, edge_costs, edge_prob, edge_interdicted], dim=-1)  # [B, edges, 4]
            
            combined = th.cat([cont_features, dep_emb, arr_emb], dim=-1)
        else:
            cont_features = th.stack([edge_capacity, edge_costs, edge_prob], dim=-1)  # [B, edges, 3]
            binary_emb = self.binary_embed(edge_interdicted)  # [B, edges, 8]
        
            combined = th.cat([cont_features, binary_emb, dep_emb, arr_emb], dim=-1)

        # Project features
        edge_embeddings = self.edge_proj(combined)  # [B, edges, 128]
        
        # Flatten and add budget
        batch_size = edge_embeddings.shape[0]
        flattened = edge_embeddings.view(batch_size, -1)
        budget = budget.reshape(-1,1)
        
        return th.cat([flattened, budget], dim=-1)

class SimpleFeatureExtractorLSTM(BaseFeaturesExtractor):                   
    def __init__(self, observation_space, num_edges=60, num_nodes=27, 
                 embedding_dim=128, lstm_hidden_dim=256,
                 edge_capacity_mean=50, edge_capacity_std=28.868,
                 edge_cost_mean=4, edge_cost_std= 0.577,
                 budget_mean = 12.5, budget_std= 7.217,  #20, budget_std= 11.547,
                 multiple_interdiction_attempts = True,
                 edge_interdicted_mean = 5, edge_interdicted_std= 2.889,):
        super().__init__(observation_space, features_dim=lstm_hidden_dim + 1)

        #Register normalization parameters as buffers
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
        
        self.num_edges = num_edges
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes
        self.lstm_hidden_dim = lstm_hidden_dim

        # Original feature processors
        self.binary_embed = nn.Embedding(2, 8)
        self.node_embed = nn.Embedding(self.num_nodes, 16)
        
        self.edge_proj = nn.Sequential(
            nn.Linear(3 + 8 + 16 + 16, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )

        # Add LSTM component
        self.lstm = nn.LSTM(
            input_size=embedding_dim + 1,  # +1 for budget per edge
            hidden_size=lstm_hidden_dim,
            batch_first=True
        )

    def forward(self, observations):
        # Truncate all edge-related features to num_edges
        def truncate(tensor):
            return tensor[..., :self.num_edges]  # Slice along last dimension
        
        # Process and truncate features
        ## Original feature processing
        edge_capacity = truncate(th.as_tensor(observations['edge_capacity'], dtype=th.float32))
        edge_capacity = (edge_capacity - self.edge_capacity_mean) / (self.edge_capacity_std + 1e-8)

        edge_costs = truncate(th.as_tensor(observations['edge_costs'], dtype=th.float32))
        edge_costs = (edge_costs - self.edge_cost_mean) / (self.edge_cost_std + 1e-8)

        edge_prob = truncate(th.as_tensor(observations['edge_interdiction_probability'], dtype=th.float32))

        if self.multiple_interdiction_attempts:
            edge_interdicted = truncate(th.as_tensor(observations['edge_interdicted'], dtype=th.float32))
            edge_interdicted = (edge_interdicted - self.edge_interdicted_mean) / (self.edge_interdicted_std + 1e-8)
        else:
            edge_interdicted = truncate(th.as_tensor(observations['edge_interdicted'], dtype=th.long))
        
        budget = th.as_tensor(observations['budget'], dtype=th.float32)
        budget = (budget - self.budget_mean) / (self.budget_std + 1e-8)
        
        dep_nodes = truncate(th.as_tensor(observations['edge_departure_node'], dtype=th.long))
        arr_nodes = truncate(th.as_tensor(observations['edge_arrival_node'], dtype=th.long))

        # Feature processing
        cont_features = th.stack([edge_capacity, edge_costs, edge_prob], dim=-1)
        binary_emb = self.binary_embed(edge_interdicted)
        dep_emb = self.node_embed(dep_nodes)
        arr_emb = self.node_embed(arr_nodes)
        combined = th.cat([cont_features, binary_emb, dep_emb, arr_emb], dim=-1)
        edge_embeddings = self.edge_proj(combined)  # [B, num_edges, embedding_dim]

        # Prepare LSTM input
        batch_size = edge_embeddings.size(0)
        
        # Expand budget to match each edge and concatenate
        budget_expanded = budget.view(batch_size, 1, 1).expand(-1, self.num_edges, 1)
        lstm_input = th.cat([edge_embeddings, budget_expanded], dim=-1)  # [B, num_edges, embedding_dim+1]

        # Process through LSTM
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        
        # Use final hidden state as features
        features = h_n[-1]  # Take last layer's hidden state [B, lstm_hidden_dim]
        
        # Reshape budget to 2D: [batch_size, 1]
        budget = budget.view(-1, 1)
    
        return th.cat([features, budget], dim=-1)  # Correct concatenation# [B, lstm_hidden_dim+1]

class SimpleFeatureExtractorLSTMv2(BaseFeaturesExtractor): #Put budget in cell state of LSTM
    def __init__(self, observation_space, num_edges=60, num_nodes=27,
                 embedding_dim=128, lstm_hidden_dim=256,
                 edge_capacity_mean=50, edge_capacity_std=28.868,
                 edge_cost_mean=4, edge_cost_std= 0.577,
                 budget_mean = 25, budget_std= 14.434):
        super().__init__(observation_space, features_dim=lstm_hidden_dim)

        #Register normalization parameters as buffers
        self.register_buffer('edge_capacity_mean', th.tensor(edge_capacity_mean))
        self.register_buffer('edge_capacity_std', th.tensor(edge_capacity_std))

        self.register_buffer('edge_cost_mean', th.tensor(edge_cost_mean))
        self.register_buffer('edge_cost_std', th.tensor(edge_cost_std))

        self.register_buffer('budget_mean', th.tensor(budget_mean))
        self.register_buffer('budget_std', th.tensor(budget_std))
        
        self.num_edges = num_edges
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes
        self.lstm_hidden_dim = lstm_hidden_dim

        # Original feature processors
        self.binary_embed = nn.Embedding(2, 8)
        self.node_embed = nn.Embedding(self.num_nodes, 16)
        
        self.edge_proj = nn.Sequential(
            nn.Linear(3 + 8 + 16 + 16, embedding_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(embedding_dim)
        )

        # Add LSTM component
        self.lstm = nn.LSTM(
            input_size=embedding_dim,  # +1 for budget per edge
            hidden_size=lstm_hidden_dim,
            batch_first=True
        )

        # Budget projection for cell state initialization
        self.budget_proj = nn.Linear(1, lstm_hidden_dim)

    def forward(self, observations):
        # Original feature processing
        edge_capacity = th.as_tensor(observations['edge_capacity'], dtype=th.float32)
        edge_capacity = (edge_capacity - self.edge_capacity_mean) / (self.edge_capacity_std + 1e-8)

        edge_costs = th.as_tensor(observations['edge_costs'], dtype=th.float32)
        edge_costs = (edge_costs - self.edge_cost_mean) / (self.edge_cost_std + 1e-8)
        
        edge_prob = th.as_tensor(observations['edge_interdiction_probability'], dtype=th.float32)
        edge_interdicted = th.as_tensor(observations['edge_interdicted'], dtype=th.long)
        
        budget = th.as_tensor(observations['budget'], dtype=th.float32)
        budget = (budget - self.budget_mean) / (self.budget_std + 1e-8)
        
        dep_nodes = th.as_tensor(observations['edge_departure_node'], dtype=th.long)
        arr_nodes = th.as_tensor(observations['edge_arrival_node'], dtype=th.long)

        # Feature processing (unchanged)
        cont_features = th.stack([edge_capacity, edge_costs, edge_prob], dim=-1)
        binary_emb = self.binary_embed(edge_interdicted)
        dep_emb = self.node_embed(dep_nodes)
        arr_emb = self.node_embed(arr_nodes)
        combined = th.cat([cont_features, binary_emb, dep_emb, arr_emb], dim=-1)
        edge_embeddings = self.edge_proj(combined)  # [B, num_edges, embedding_dim]

        # Initialize cell state with budget information
        batch_size = edge_embeddings.size(0)
        c_0 = self.budget_proj(budget.view(-1, 1))  # [B, hidden_dim]
        c_0 = c_0.unsqueeze(0).expand(1, batch_size, self.lstm_hidden_dim)  # [1, B, hidden_dim]
        h_0 = th.zeros_like(c_0)  # Default hidden state

        # Process through LSTM
        lstm_out, (h_n, c_n) = self.lstm(edge_embeddings, (h_0,c_0))
    
        return h_n[-1]  # [B, lstm_hidden_dim]

#End custom Features Extractor

# Custom policy network architecture
policy_kwargs = dict(
    features_extractor_class=SimpleFeatureExtractor, # SimpleFeatureExtractorLSTM, #
    features_extractor_kwargs={
        'num_edges': 17, #40, #168, #  270, #60, #
        'num_nodes': 14, #27,  #102, #27, #
        'embedding_dim': 64, #128,
        'multiple_interdiction_attempts': multiple_interdiction_attempts,
       # "lstm_hidden_dim": 256
    },
#    net_arch=dict(pi=[1024,1024,512,256], vf=[1024,1024,512,256]),  # Post-feature-extractor layers
#    net_arch=dict(pi=[5000,1600,512,512], vf=[5000, 1000, 200, 100]),  # Post-feature-extractor layers - PPO and A #original 512,512,256

#    net_arch=dict(pi=[512,512,512, 512], vf=[512,256,128,64]), #G5x5_PPO_EX006A, _MaskablePPO_EX005A
#    net_arch=dict(pi=[7680,7680,7680, 7680], vf=[7680,7680,7680, 7680]), #G5x5_MaskablePPO_EX005B&1
#    net_arch=dict(pi=[1024,1024,1024, 512], vf=[1024,1024,512,256]), #G5x5_MaskablePPO_EX005C&1
#    net_arch=dict(pi=[512,512,512, 512], vf=[512,256,128,64]), #G5x5_PPO_EX006A _A2C_EX006A
#     net_arch=dict(pi=[512,512,512,512,512,512,512], vf=[512,256,128,64,32,16,8]), #G8x8_PPO_EX006A
#     net_arch=dict(pi=[1548,1548,1548,1548,1548,1548,1548], vf=[1548,774,387,194,97,48,24]), #G8x8_PPO_EX006A

#    net_arch=dict(pi=[10836], vf=[10836]),
#    net_arch=dict(pi=[4096,2048, 1024,512,256], vf=[4096,2048, 1024,512,256]),  # Post-feature-extractor layers - PPO
#    net_arch=[512,512,512,512], #G5x5_DQN_EX006A,M
     net_arch=[512,512],
#    net_arch=[4096,2048,1024,512],   # Version B:[512, 512, 512, 512], #DQN
    activation_fn=nn.LeakyReLU,
#    ortho_init=True  # Enable orthogonal initialization
)

class ReplayBufferSaveCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            buffer_file = os.path.join(self.save_path, f"{model_name}_replay_buffer.pkl")
            with open(buffer_file, "wb") as f:
                pickle.dump(self.model.replay_buffer, f)
        return True



if __name__ == "__main__":
    num_envs = n_cpus  # Number of environments
    envs = [make_env for _ in range(num_envs)]
    vec_env = SubprocVecEnv(envs)
    #vec_env = VecNormalize(SubprocVecEnv(envs), norm_obs=False)  #keys = ["edge_capacity", "edge_costs", "budget"]

    # Create evaluation environment (not vectorized)
    eval_env = DummyVecEnv([  # Single environment (not vectorized)
        lambda: Monitor(
            ce.CustomEnv(
                nodes, edges, deterministic_agent=deterministicOutcomes, 
                      fixed_costs=fixedCosts, curriculum_training=False, min_training_budget = min_training_budget, max_training_budget = max_training_budget,
                        multiple_interdiction_attempts=multiple_interdiction_attempts
            )
        )
    ])
    
    if agent == "MaskablePPO":
        # First define a mask function that calls your environment's action_mask method
        def mask_fn(env):
            return env.unwrapped.action_mask() #env.action_mask()

        eval_callback = MaskableEvalCallback(  # Changed from EvalCallback
                                 eval_env,
                                 best_model_save_path=f"{models_dir}/{model_name}",
                                 log_path=f"{models_dir}/{model_name}",
                                 eval_freq=700, #max(384000 // num_envs, 1),
                                 n_eval_episodes=576,
                                 deterministic=True,
                                 render=False,
                                 verbose=False)
    else:
        eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=f"{models_dir}/{model_name}",
                                 log_path=f"{models_dir}/{model_name}",
                                 eval_freq=700,# 350 for PPO #320 for DQN, 
                                 n_eval_episodes=576, #was 288
                                 deterministic=True,
                                 render=False,
                                 verbose=False)
    
    #eval_env = VecNormalize(eval_env, norm_obs=False) #keys = ["edge_capacity", "edge_costs", "budget"]
    #eval_env.training = False
    #eval_env.norm_reward = False

    if agent == "A2C":
        from stable_baselines3 import A2C
        model = A2C("MultiInputPolicy", vec_env, verbose=0, n_steps=10, 
                    ent_coef=0.01,
                    learning_rate=linear_schedule(initial_learning_rate),
                    policy_kwargs=policy_kwargs)
    
    elif agent == "DQN":
        from stable_baselines3 import DQN
        model = DQN("MultiInputPolicy", vec_env, verbose=0, buffer_size=3000000, learning_starts=100000,
                    batch_size=720,
                    train_freq=5, #10
                    target_update_interval=10000,
                    gradient_steps=2,  #4
                    exploration_fraction=0.6, exploration_initial_eps=1.0, exploration_final_eps=0.05,
                    policy_kwargs=policy_kwargs, 
                    learning_rate=linear_schedule(initial_learning_rate))
        replay_buffer_callback = ReplayBufferSaveCallback(save_freq=1750, save_path=f"{models_dir}/{model_name}")
        
    elif agent == "PPO":
        from stable_baselines3 import PPO
        model = PPO(policy="MultiInputPolicy", env=vec_env, verbose=0, 
                    learning_rate= linear_schedule(initial_learning_rate), #added learning rate schedule
                    n_steps = 35, #EX4 () and again for EX5 (768)
                    n_epochs = 15, #was 10
                    ent_coef=0.05, #added
                    batch_size=5040, #increased from 128
                    gamma = .999,
                    policy_kwargs=policy_kwargs) #EX2 had learning_rate = 0.0001
        
    elif agent == "MaskablePPO":
        from sb3_contrib import MaskablePPO
        model = MaskablePPO(
            policy="MultiInputPolicy", env=vec_env, verbose=0,
            learning_rate=linear_schedule(initial_learning_rate),
            n_steps=35, #384000 // num_envs,
            n_epochs=15,
            ent_coef=0.05,
            batch_size=5040,
            gamma=0.999,
            policy_kwargs=policy_kwargs)

    # Initialize Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=1750, 
                                             save_path = f"{models_dir}/{model_name}",
                                             #save_vecnormalize=True,  # Save normalization stats
                                             name_prefix=model_name)
   
    # Train the agent
    model.learn(total_timesteps=timesteps, callback=[checkpoint_callback, eval_callback, replay_buffer_callback], #CurriculumCallback()],
                progress_bar = True)