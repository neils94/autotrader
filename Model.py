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
import torchvison
from torchvision import transforms, utils, models
from torch import optim as optim

class Actor(nn.Module):
            """
        - Params:
                - d_model(int): number of query weights for attention model to attend to
                - nhead (int): number of multiheaded attention heads
                - num_encoder_layers (int): number of mha + dimfeedforward encoder layers
                - num_decoder_layers (int): same as num_encoder_layers but for decoder
                - dropout (float): dropout rate for conv and linear layers
                - dim_feedforward (int): number of dimensions for ffn layers of the encoder/decoder blocks
                - hidden_layers 1&2 (int): number of dimensions and number of features for linear layers
                - input_channels (int): number of color channels in the conv2d input
                - output_channels (int): number of desired output channels from conv2d (subtract from input channels to get # of filters)
                - kernel_size (int or tuple): size of convolution filters, one number for square conv (ie: 3 for 3x3) for uneven convolutions specify as tuple (int,int)
                - params (callable function): parameters for Adam to update
                - betas (float64): beta hyperparameter for Adam to know how much to weight past batch of gradients relative to current (0.9, or 0.99 usually)
                - eps (float): epsilon hyperparameter for Adam to add some noise to previous batch so that it's a non-zero denominator (default= 1e-8)
                - lr (float): learning rate hyperparameter for Adam to scale size of gradients
                - wd (float): weight decay hyperpareter for Adam to scale parameters and add them to the gradient (default=1 e-4) 
        """

    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dropout,
                dim_feedforward, hidden_layer, input_channels=3,output_channels=3, kernel_size=1):

        super().__init__()
        self.conv = nn.Conv2d(input_channels,output_channels, kernel_size)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.batch_normalization = nn.BatchNorm2d()
        self.fc1 = nn.Linear(transformer_output, hidden_layer)
        self.fc2 = nn.Linear(hidden2_layer, hidden2_layer)
        self.fc3 = nn.Linear(hidden2_layer, hidden2_layer)
        self.fc4 = nn.Linear(hidden2_layer, hidden2_layer)
        self.fc5 = nn.Linear(hidden2_layer, hidden2_layer)
        self.fc6 = nn.Linear(hidden2_layer, action_size)
        self.actor_optim = optim.Adam(params, betas, eps, lr, wd)
    


    def forward(self, img: torch.Tensor):
        """TO-DO: Feed forward function for the transformer
        Params = conv_output - input tensor to forward propogate through the model"""

        conv_output = self.conv(img)
        flattened_img = conv_output.flatten(0,-1)
        transformer_output = self.transformer(flattened_img)
        x = F.relu(self.fc1(transformer_output))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        

        return F.relu(self.fc6)

    def build_optimus_prime(self):
        return torch.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1)
    
    
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_layer, hidden2_layer, seed):
        
        
    def __init__(self, d_model: int, nhead: int, num_encoder_layers: int, num_decoder_layers: int, dropout: float32,
                dim_feedforward: int, hidden_layer: int, input_channels: int,output_channels: int, kernel_size: int, params: callable, betas: float64, eps: float64, lr: float64, wd: float64):

        """
        - Params:
                - same as actor

        """

        super().__init__()
        self.conv = nn.Conv2d(input_channels,output_channels, kernel_size)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.batch_normalization = nn.BatchNorm2d()
        self.fc1 = nn.Linear(transformer_output, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden2_layer)
        self.fc3 = nn.Linear(hidden2_layer, hidden2_layer)
        self.fc4 = nn.Linear(hidden2_layer, hidden2_layer)
        self.fc5 = nn.Linear(hidden2_layer, hidden2_layer)
        self.fc6 = nn.Linear(hidden2_layer, action_size)
        self.actor_optim = optim.Adam(params, betas, eps, lr, wd)
    


    def forward(self, img: torch.Tensor):
        """TO-DO: Feed forward function for the transformer
        Params = conv_output - input tensor to forward propogate through the model"""

        conv_output = self.conv(img)
        flattened_img = conv_output.flatten(0,-1)
        transformer_output = self.transformer(flattened_img)
        x = F.relu(self.fc1(transformer_output))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        

        return F.relu(self.fc6)