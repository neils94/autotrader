from multiprocessing.sharedctypes import Value
import torch 
from torch import nn
import torch.nn.functional as F
from torch import multiprocessing as mp
import torch.nn.quantized as nnq
import numpy as np

class Actor(nn.Module):
    
    def __init__(self, state, linear_input, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
    kernel_size, kernel_size2, stride, stride2):
        
        super(Actor, self).__init__()
        self.conv = nn.Conv2d(input_channels,output_channels, kernel_size, stride)
        self.conv2 = nn.Conv2d(input_channels2,output_channels2, kernel_size2, stride2)    
        self.actor_1 = torch.nn.Linear(linear_input, hidden_layer)
        self.actor_2 = torch.nn.Linear(hidden_layer, hidden2_layer)
        self.actor_3 = torch.nn.Linear(hidden2_layer, action_size)    
        self.logsoft = torch.nn.LogSoftmax()
        self.batch_norm = torch.nn.BatchNorm2d(num_features=1)
           
    def forward(self, state):
        state = torch.unsqueeze(state, dim=0)
        state = self.batch_norm(state)
        input_state = F.relu(self.conv(state))
        conv_output = F.relu(self.conv2(input_state))
        conv_flatten = conv_output.flatten(start_dim=0, end_dim=-1)
        x = F.relu(self.actor_1(conv_flatten))
        x = F.relu(self.actor_2(x))
        action_probs = F.softmax(self.actor_3(x), dim=-1)
        
        return action_probs



class Critic(nn.Module):
    def __init__(self, state, linear_input, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
    kernel_size, kernel_size2, stride, stride2):
        
        super(Critic, self).__init__()
        self.conv = nn.Conv2d(input_channels,output_channels, kernel_size, stride)
        self.conv2 = nn.Conv2d(input_channels2,output_channels2, kernel_size2, stride2)
        self.critic_1 = torch.nn.Linear(linear_input, hidden_layer)
        self.critic_2 = torch.nn.Linear(hidden_layer + action_size, hidden2_layer)
        self.critic_3 = torch.nn.Linear(hidden2_layer, 1)      
        self.batch_norm = torch.nn.BatchNorm2d(num_features=1)
           
    def forward(self, state, actions):
        state = torch.unsqueeze(state, dim=0)
        state = self.batch_norm(state)

        input_state = F.relu(self.conv(state))
        conv_output = F.relu(self.conv2(input_state))
        y = F.relu(self.critic_1(conv_output))
        y = torch.cat([y, actions], dim=1)
        y = F.relu(self.critic_2(y))
        state_value = F.relu(self.critic_3(y))

        return state_value
