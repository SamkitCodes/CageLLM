# modified from https://github.com/geekyutao/PyTorch-PPO/blob/master/PPO_discrete.py

import torch, torch.nn as nn
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states, self.actions = [], []
        self.rewards, self.is_terminals = [], []
        self.logprobs, self.next_states = [], []

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(ActorCritic, self).__init__()
        
        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def act(self, state, memory, deterministic=False):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action = torch.argmax(action_probs, dim=1) if deterministic else dist.sample()
        action_logp = dist.log_prob(action)
        
        memory.states.append(state.detach())
        memory.actions.append(action.detach())
        memory.logprobs.append(action_logp.detach())
        
        return action
    
    def evaluate(self, states, actions):
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        
        state_values = self.critic(states).squeeze(-1)
        
        return action_logprobs, state_values, dist_entropy
