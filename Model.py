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
        self.dropout = torch.nn.Dropout(0.2, inplace=True)
        self.batch_norm = torch.nn.BatchNorm2d(num_features=3)

    def he_init(self, model):
        if isinstance(model, nn.Linear):
            torch.nn.init.kaiming_uniform_(model.weight)
            model.bias.data.fill_(0.01)

    def forward(self, state):
        state = torch.unsqueeze(state, dim=0)
        state = self.batch_norm(state)
        self.apply(self.he_init)

        input_state = F.relu(self.dropout(self.conv(state)))
        conv_output = F.relu(self.dropout(self.conv2(input_state)))
        conv_flatten = conv_output.flatten(start_dim=0, end_dim=-1)
        x = F.relu(self.dropout(self.actor_1(conv_flatten)))
        x = F.relu(self.dropout(self.actor_2(x)))
        x = F.relu(self.actor_3(x))
        
        return F.softmax(x, dim=-1)


class Critic(nn.Module):
    def __init__(self, state, linear_input, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
    kernel_size, kernel_size2, stride, stride2):
        
        super(Critic, self).__init__()
        self.conv = nn.Conv2d(input_channels,output_channels, kernel_size, stride)
        self.conv2 = nn.Conv2d(input_channels2,output_channels2, kernel_size2, stride2)
        self.critic_1 = torch.nn.Linear(linear_input, hidden_layer)
        self.critic_2 = torch.nn.Linear(hidden_layer + action_size, hidden2_layer)
        self.critic_3 = torch.nn.Linear(hidden2_layer, 1)      
        self.dropout = torch.nn.Dropout(0.2, inplace=True)
        self.batch_norm = torch.nn.BatchNorm2d(num_features=1)

    def he_init(self, model):
        if isinstance(model, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(model.weight)
            model.bias.data.fill_(0.01)

    def forward(self, state, actions):
        state = torch.unsqueeze(state, dim=0)
        state = self.batch_norm(state)
        self.apply(self.he_init)

        input_state = F.relu(self.dropout(self.conv(state)))
        conv_output = F.relu(self.dropout(self.conv2(input_state)))
        conv_flatten = conv_output.flatten(start_dim=0, end_dim=-1)
        y = F.relu(self.dropout(self.critic_1(conv_flatten)))
        y = torch.cat([y, actions], dim=1)
        y = F.relu(self.critic_2(y))
        state_value = F.relu(self.critic_3(y))

        return state_value


#build a DQN model with same architecture 

class DQN(nn.Module):
    def __init__(self, state, linear_input, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
    kernel_size, kernel_size2, stride, stride2):
        super(DQN, self).__init__()
        self.conv = nn.Conv2d(input_channels,output_channels, kernel_size, stride)
        self.conv2 = nn.Conv2d(input_channels2,output_channels2, kernel_size2, stride2)    
        self.fc1 = torch.nn.Linear(linear_input, hidden_layer)
        self.fc2 = torch.nn.Linear(hidden_layer, hidden2_layer)
        self.fc3 = torch.nn.Linear(hidden2_layer, action_size) 

        self.batch_norm = torch.nn.BatchNorm2d(num_features=1)
           
    def forward(self, state):
        state = torch.unsqueeze(state, dim=0)
        state = self.batch_norm(state)

        x = F.relu(self.conv(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x

class ResNetBlock(nn.Module):
    def __init__(self, input_channels, input_channels1, output_channels, output_channels1, kernel_size, kernel_size1, stride):
        super(ResNetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channels, output_channels, kernel_size, stride)
        self.conv2 = torch.nn.Conv2d(input_channels1, output_channels1, kernel_size1, stride)
        self.batch_norm = torch.nn.BatchNorm2d(num_features=1)

    def he_init(self, model):
        if isinstance(model, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(model.weight)
            model.bias.data.fill_(0.01)
            self.apply(self.he_init)

    
    def forward(self, input):
        x = self.conv1(input)
        x = self.batch_norm(x)
        x = F.relu(self.conv2(x))

        return x
