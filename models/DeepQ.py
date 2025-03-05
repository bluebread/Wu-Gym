import torch
from torch import nn
import numpy as np
import gymnasium
import random
from collections import namedtuple
import math

class SimpleNetwork(nn.Module):
    def __init__(self, num_actions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_relu_stack = nn.Sequential(
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.LazyLinear(16),
            nn.ReLU(),
            nn.LazyLinear(num_actions)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

class NaiveDeepQNetwork():
    def __init__(self, env: gymnasium.Env, *args, **kwargs):
        self.ActType = env.action_space.dtype
        self.ObsType = env.observation_space.dtype
        self.num_actions = env.action_space.n
        self.qnet = SimpleNetwork(self.num_actions)
        self.target_qnet = SimpleNetwork(self.num_actions)
        self.gamma = 0.999
        self.update_delay = 50
        self.epsilon = 0.1
        self.learning_rate = 1e-4
        self.update_count = 0

    def take_action(self, obs):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        self.qnet.eval()

        with torch.no_grad():    
            obs = torch.tensor(obs)
            qvalues = self.qnet(obs)

            return qvalues.argmax().item()
    
    def train(self, transitions):
        Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'final'])
        batch = Transition(*zip(*transitions))
        batch_size = len(transitions)

        states = torch.tensor(np.stack(batch.state))
        actions = torch.tensor(np.stack(batch.action))
        rewards = torch.tensor(np.stack(batch.reward)).float()
        next_states = torch.tensor(np.stack(batch.next_state))
        final_mask = torch.tensor(np.stack(batch.final))

        qvalues = self.qnet(states).gather(1, actions.unsqueeze(dim=1))
        next_vvalues = torch.zeros(batch_size)
        non_final_next_states = next_states[~final_mask]

        with torch.no_grad():
            next_vvalues[~final_mask] = self.target_qnet(non_final_next_states).max(dim=1).values
        
        expect_qvalues = rewards + self.gamma * next_vvalues

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.qnet.parameters(), lr=self.learning_rate)

        loss = criterion(qvalues, expect_qvalues)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.update_count += 1

        if self.update_count % self.update_delay == 0:
            self.target_qnet = self.qnet
