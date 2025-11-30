import torch
import torch.nn as nn
import torch.distributions as distributions
import numpy as np
from src.utils import layer_init

class PPO(nn.Module):
    def __init__(self, input_dims, hidden_dims, action_dims, lr=3e-4):
        super().__init__()

        self.backbone = nn.Sequential(
            layer_init(nn.Linear(input_dims, hidden_dims)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dims, hidden_dims)),
            nn.Tanh()
        )

        self.actor_mean = layer_init(nn.Linear(hidden_dims, action_dims), std=0.01)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dims))

        self.critic_head = layer_init((nn.Linear(hidden_dims, 1)), std = 1)
    
    def get_value(self, x):
        features = self.backbone(x)
        value = self.critic_head(features)

        return value.reshape(-1)
    
    def get_action_and_value(self, x, action = None):
        features = self.backbone(x)

        mean = self.actor_mean(features)
        std = self.actor_log_std.exp()

        dist = distributions.Normal(mean, std)

        if action is None:
            action = dist.sample()

        value = self.critic_head(features)

        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), value.reshape(-1)