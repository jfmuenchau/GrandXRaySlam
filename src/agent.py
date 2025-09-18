import torch
import torch.nn as nn
from .utils import ValueMemoryEMA, ValueMemory, augmentation_space


class Actor(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden)
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.alpha_head = nn.Linear(hidden, out_features)
        self.beta_head = nn.Linear(hidden, out_features)

    def forward(self, x):
        x = torch.relu(self.linear1(self.layer_norm1(x)))
        x = torch.relu(self.linear2(x))
        return torch.softmax(self.alpha_head(x), dim=-1) + 1, torch.softmax(self.beta_head(x), dim=-1) + 1

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
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(self.layer_norm1(x)))
        x = torch.relu(self.linear2(x))
        return self.head(x)


class Controller(nn.Module):

    def __init__(self, in_features, hidden):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden)
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, len(augmentation_space))

    def forward(self, x):
        x = torch.relu(self.linear1(self.layer_norm1(x)))
        x = torch.relu(self.linear2(x))
        return self.head(x)
    
    def get_dist(self, x):
        out = self(x)
        dist = torch.distributions.Categorical(logits=out)
        return dist


class Agent(nn.Module):

    def __init__(self, in_features, control=False, actor=False):
        super().__init__()
        self.val_memory = ValueMemory()
        self.loss_memory = ValueMemory()

        self.critic = Critic(in_features=in_features, hidden=128)

        self.store_ = {}

        self.control_ = control
        self.actor_ = actor
        if control:
            self.controller = Controller(in_features=in_features, hidden=128)

        if actor:
            self.actor = Actor(in_features=in_features, hidden=128, out_features=1) 

        self.actor_optimizer = torch.optim.Adam(
            params=self.actor.parameters(), lr=3e-5, weight_decay=5e-4
        )
        self.critic_optimizer = torch.optim.Adam(
            params=self.critic.parameters(), lr=3e-5, weight_decay=5e-4
        )

    def action(self, state):
        action_actor = None
        action_controller = None
        if self.actor_:  
            dist = self.actor.get_dist(state.detach())
            action_actor = dist.sample()
            self.store_["action_actor"] = action_actor
            self.store_["dist_actor"] = dist
        if self.control_:
            dist = self.controller.get_dist(state.detach())
            action_controller = dist.sample()
            self.store_["action_controller"] = action_controller
            self.store_["dist_controller"] = dist

        return action_actor, action_controller

    def update(self, key, state, reward):
        value = self.critic(state)
        prev_state = self.val_memory(key, state)
        prev_value = self.critic(prev_state)
        
        with torch.no_grad():
            td_target = reward + 0.99 * value

        if self.actor_ and self.control_:
            dist_actor, action_actor = self.store_["dist_actor"], self.store_["action_actor"]
            dist_control, action_control = self.store_["dist_controller"], self.store_["action_controller"]

            log_prob_actor = dist_actor.log_prob(action_actor)
            log_prob_control = dist_control.log_prob(action_control)

            log_prob = log_prob_actor + log_prob_control  

        elif self.actor_:
            dist, action = self.store_["dist_actor"], self.store_["action_actor"]
            log_prob = dist.log_prob(action)

        elif self.control_:
            dist, action = self.store_["dist_controller"], self.store_["action_controller"]
            log_prob = dist.log_prob(action)

        actor_loss = -(log_prob * (td_target - prev_value.detach())).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        critic_loss = torch.nn.functional.mse_loss(td_target, prev_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss, critic_loss