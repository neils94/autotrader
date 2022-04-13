"Model"

import tensorflow as tf 
import tensorflow.layers as tfl
import numpy as np
from tensorflow.keras.layers import Activations
import matplotlib.pyplot as plt


#class DETR_model():
    #def __init__(self):
        #initialize DETR model global variables
        

    #def CNN
    
    #def transformer


class Actor(nn.Module):
    
    def __init__(self, state_size, action_size, hidden_layer, hidden2_layer, seed):
        
        super(Actor, self).__init__()
        self.fc1 = tfl.layers.Linear(num_input_dim=state_size, units=hidden_layer)
        self.fc2 = tfl.layers.Linear(num_input_dim=state_size, units=hidden_layer)
        self.fc3 = tfl.layers.Linear(num_input_dim=state_size, units=hidden_layer)
        self.fc4 = tfl.layers.Linear(num_input_dim=state_size, units=hidden_layer)
        self.fc5 = tfl.layers.Linear(num_input_dim=state_size, units=hidden_layer)
        self.fc6 = tfl.layers.Linear(num_input_dim=state_size, units=action_size)
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=1)(state_size)
           
    def forward(self, states):
        if states.dim() == 1:
            states = tf.expand_dims(states, 0)
        states = self.batch_norm(states)
        
        relu = tensorflow.keras.layers.Activations('relu')
        x = relu(self.fc1(self.fc1))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = relu(self.fc4(x))
        x = relu(self.fc5(x))
        x = relu(self.fc1(x))
        
        
        
        return self.fc6(x)
    
    
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_layer, hidden2_layer, seed):
        
        
        super(Critic,self).__init__()
        self.fc1 = tfl.layers.Linear(num_input_dim=state_size, units=hidden_layer)
        self.fc2 = tfl.layers.Linear(num_input_dim=hidden_layer, units=hidden_layer2)
        self.fc3 = tfl.layers.Linear(num_input_dim=hidden_layer, units=hidden_layer2)
        self.fc4 = tfl.layers.Linear(num_input_dim=hidden_layer, units=hidden_layer2)
        self.fc5 = tfl.layers.Linear(num_input_dim=hidden_layer, units=hidden_layer2)
        self.fc6 = tfl.layers.Linear(num_input_dim=hidden_layer, units=action_size)
        
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=1)(state_size)
        
    
    def forward(self, states):
        if states.dim() == 1:
            states = tf.expand_dims(states, 0)
            
        states = self.batch_norm(states)

        relu = tensorflow.keras.layers.Activations('relu')
        xs = relu(self.fc1(self.fc1))
        x = relu(self.fc2(xs))
        x = relu(self.fc3(x))
        x = relu(self.fc4(x))
        x = relu(self.fc5(x))
        x = relu(self.fc1(x))
        
        return self.fc6(x)
