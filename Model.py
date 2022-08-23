from multiprocessing.sharedctypes import Value
import torch 
from torch import nn
import torch.functional as F


@torch.jit.script
class Actor(nn.Module):
    
    def __init__(self, state, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
    kernel_size, kernel_size2, stride, stride2):
        
        super(Actor, self).__init__()
        self.conv = nn.Conv2d(input_channels,output_channels, kernel_size, stride)
        self.conv2 = nn.Conv2d(input_channels2,output_channels2, kernel_size2, stride2)    
        self.fc1 = torch.nn.Linear(state, hidden_layer)
        self.fc2 = torch.nn.Linear(hidden_layer, hidden2_layer)
        self.fc3 = torch.nn.Linear(hidden2_layer, action_size) 
        self.batch_norm = torch.nn.BatchNorm2d(state)
           
    def forward(self, state):

        state = self.batch_norm(state)

        x = F.relu(self.conv(state))
        x = self.conv2(x)
        x = self.transformer(src=x,target=x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x))
        
        
        
        return action_probs
    
@torch.jit.script    
class Critic(nn.Module):
    def __init__(self, state, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, 
    output_channels2, kernel_size, kernel_size2, stride, stride2):
        
        
        super(Critic,self).__init__()
        self.conv = nn.Conv2d(input_channels,output_channels, kernel_size, stride)
        self.conv2 = nn.Conv2d(input_channels2, output_channels2, kernel_size2, stride2)   
        self.fc1 = torch.nn.Linear(state, hidden_layer)
        self.fc2 = torch.nn.Linear(hidden_layer + action_size, hidden2_layer)
        self.fc3 = torch.nn.Linear(hidden2_layer, 1) 
        self.batch_norm = torch.nn.BatchNorm2d(state)
        
    
    def forward(self, state, actions):

        state = self.batch_norm(state)

        x = F.relu(self.conv(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x)) 
        x = torch.cat([x, actions], dim=1)  
        x = F.relu(self.fc2(x))
        value = self.fc3(x)

        return value
