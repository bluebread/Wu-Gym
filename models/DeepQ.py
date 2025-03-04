import torch
from torch import nn
import numpy as np
import gymnasium
import random

class SimpleNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_relu_stack = nn.Sequential(
            # nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(4),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
def ReplayBufferOneStep():
    def __init__(self, max_size, batch_size=64):
        super().__init__()
        self.buffer = []
        self.max_size = max_size
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state):
        if (len(self.buffer) > self.buffer_size):
            self.buffer.pop(0)
        self.buffer.append(tuple(state, action, reward, next_state))
    
    def sample_batch(self):
        return random.sample(self.buffer, self.batch_size)
    

class DeepQNetwork():
    def __init__(self, env: gymnasium.Env):
        self.ActType = env.action_space.dtype
        self.ObsType = env.observation_space.dtype
        self.qnet = SimpleNetwork()
        self.target_qnet = SimpleNetwork()
        self.replay_buffer = ReplayBufferOneStep()

    def take_action(self, obs):
        obs = torch.tensor(obs)
        ap = self.qnet(obs)
        return ap.argmax().item()
    
    def record_experience(self, s, a, r, next_s):
        self.replay_buffer.add(s, a, r, next_s)
    
    # def train(self):
    #     batch_features = self.replay_buffer.sample_batch()