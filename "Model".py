"Model"

import torch 
from torch import nn
import torch.functional as F
import numpy as np
import matplotlib.pyplot as plt


#class DETR_model():
    #def __init__(self):
        #initialize DETR model global variables
        

    #def CNN
    
    #def transformer

@torch.jit.script
class Actor(nn.Module):
    
    def __init__(self, state_size, action_size, hidden_layer, hidden2_layer, seed):
        
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, hidden_layer)
        self.fc2 = torch.nn.Linear(hidden_layer, hidden2_layer)
        self.fc3 = torch.nn.Linear(hidden_layer, hidden2_layer)
        self.fc4 = torch.nn.Linear(hidden_layer, hidden2_layer)
        self.fc5 = torch.nn.Linear(hidden_layer, hidden2_layer)
        self.fc6 = torch.nn.Linear(hidden2_layer, action_size)
        
        self.batch_norm =torch.nn.BatchNorm1d(state_size)
           
    def forward(self, states):
        if states.dim() == 1:
            states = torch.unsqueeze(states, 0)
        states = self.batch_norm(states)
        
        x = F.relu(self.fc1(self.fc1))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        
        
        
        return self.fc6(x)
    
@torch.jit.script    
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_layer, hidden2_layer, seed):
        
        
        super(Critic,self).__init__()
        self.fc1 = torch.nn.Linear(state_size, hidden_layer)
        self.fc2 = torch.nn.Linear(hidden_layer, hidden2_layer)
        self.fc3 = torch.nn.Linear(hidden_layer, hidden2_layer)
        self.fc4 = torch.nn.Linear(hidden_layer, hidden2_layer)
        self.fc5 = torch.nn.Linear(hidden_layer, hidden2_layer)
        self.fc6 = torch.nn.Linear(hidden2_layer, action_size)
        
        self.batch_norm = torch.nn.BatchNorm1d(state_size)
        
    
    def forward(self, states):
        if states.dim() == 1:
            states = torch.unsqueeze(states, 0)
            
        states = self.batch_norm(states)

        xs = F.relu(self.fc1(self.fc1))
        x = F.relu(self.fc2(xs))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        
        return self.fc6(x)
