"Model"

import torch
from torch import nn
import numpy as np


import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim

import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


#class Vision_model():
    #def __init__(self):
        #pass

    #def 


class Actor(nn.Module):
    
    def __init__(self, state_size, action_size, hidden_layer, hidden2_layer, seed):
        
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(0)
        self.fc1 = nn.Linear(state_size, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden2_layer)
        self.fc3 = nn.Linear(hidden_layer, hidden2_layer)
        self.fc4 = nn.Linear(hidden_layer, hidden2_layer)
        self.fc5 = nn.Linear(hidden_layer, hidden2_layer)
        self.fc6 = nn.Linear(hidden2_layer, action_size)
    
           
    def forward(self, states):
        if states.dim() == 1:
            states = torch.unsqueeze(states, 0)
        states = self.batch_norm(states)
        
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        
        
        
        return self.fc6(x)
    
    
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_layer, hidden2_layer, seed):
        
        
        super(Critic,self).__init__()
        self.seed = torch.manual_seed(0)
        self.fc1 = nn.Linear(state_size, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer + action_size, hidden2_layer)
        self.fc3 = nn.Linear(hidden_layer + action_size, hidden2_layer)
        self.fc4 = nn.Linear(hidden_layer + action_size, hidden2_layer)
        self.fc5 = nn.Linear(hidden_layer + action_size, hidden2_layer)
        self.fc6 = nn.Linear(hidden2_layer, 1)
        
        self.batch_norm = nn.BatchNorm1d(state_size)
        
    
    def forward(self, states):
        if states.dim() == 1:
            states = torch.unsqueeze(states, 0)
            
        states = self.batch_norm(states)

        xs = F.relu(self.fc1(states))
        x = F.relu(self.fc1(xs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        
        return self.fc6(x)