from collections import namedtuple, deque
import random
import torch
import numpy as np
from torch import device
 
class ReplayBuffer():
    def __init__(self, buffer_size, batch_size):
        self.experience = namedtuple(typename = "experience", field_names=["states", "actions", "action_probs", "rewards", "next_states", "close"])
        self.memory = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        
    def add(self, states, actions, action_probs, rewards, next_states, close):
        exp = self.experience(states, actions, action_probs, rewards, next_states, close)
        self.memory.append(exp)
        
        
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack(exp.states for exp in experiences if exp is not None)).float().to(device)
        actions = torch.from_numpy(np.vstack(exp.actions for exp in experiences if exp is not None)).float().to(device)
        action_probs = torch.from_numpy(np.vstack(exp.action_probs for exp in experiences if exp is not None)).float().to(device)
        rewards = torch.from_numpy(np.vstack(exp.rewards for exp in experiences if exp is not None)).float().to(device)
        next_states = torch.from_numpy(np.vstack(exp.next_states for exp in experiences if exp is not None)).float().to(device)
        close = torch.from_numpy(np.vstack(exp.dones for exp in experiences if exp is not None).astype(np.uint8)).float().to(device)
        
        return (states, actions, action_probs, rewards, next_states, close)
    
    def __len__(self):
        return len(self.memory)
