import torch
import torch.nn as nn
from .utils import ValueMemoryEMA, augmentation_space


class Actor(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden)
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.layer_norm2 = nn.LayerNorm(hidden)
        self.alpha_head = nn.Linear(hidden, out_features)
        self.beta_head = nn.Linear(hidden, out_features)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.layer_norm1(x)
        x = torch.relu(self.linear2(x))
        x = self.layer_norm2(x)
        return torch.softmax(self.alpha_head(x)) + 1, torch.softmax(self.beta_head) + 1
    
    def get_dist(self, x):
        alpha, beta = self(x)
        dist = torch.distributions.Beta(alpha, beta)
        return dist


class Critic(nn.Module):

    def __init__(self, in_features, hidden):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden)
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.layer_norm2 = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.layer_norm1(x)
        x = torch.relu(self.linear2(x))
        x = self.layer_norm2(x)
        return self.head(x)


class Controller(nn.Module):

    def __init__(self, in_features, hidden):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden)
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.layer_norm2 = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, len(augmentation_space))

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.layer_norm1(x)
        x = torch.relu(self.linear2(x))
        x = self.layer_norm2(x)
        return self.head(x)
    
    def get_dist(self, x):
        out = self(x)
        dist = torch.distributions.Categorical(logits=out)
        return dist


class Agent:

    def __init__(self, in_features, control=False):
        self.val_memory = ValueMemoryEMA(alpha=0.3)
        self.loss_memory = ValueMemoryEMA(alpha=0.3)

        self.critic = Critic(in_features=in_features, hidden=128)

        self.store_ = {}

        if control:
            self.actor = Controller(in_features=in_features, hidden=128)

        else:
            self.actor = Actor(in_features=in_features, hidden=128, out_features=1) 

        self.actor_optimizer = torch.optim.Adam(
            params=self.actor.parameters(), lr=3e-5, weight_decay=5e-4
        )
        self.actor_optimizer = torch.optim.Adam(
            params=self.critic.parameters(), lr=3e-5, weight_decay=5e-4
        )

    def action(self, state):
        dist = self.actor.get_dist(state.detach())
        action = dist.sample()
        self.store_["action"] = action
        self.store_["dist"] = dist
        return action

    def update(self, key, state, sample_loss):
        dist, action = self.store_["dist"], self.store_["action"]
        
        value = self.critic(state)

        ema_value = self.val_memory.get_multi(key)

        advantage = 0.99 * value - ema_value