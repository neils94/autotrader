from Model import Actor, Critic
import random
import torch
from Environment import environment
import torch.nn.functional as F
from torch import optim
from ReplayBuffer import ReplayBuffer
import numpy as np
from numpy import random


lr = 1e-4
wd = 1e-4
lamb = 1e-2
gamma = 0.99
tau = 1e-3
buffer_size = int(1e5)
batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0
update_every = 16
eps= 0.8



class Agent():
    """
    -  
    """
    
    def __init__(self, batch_size, buffer_size, state_size, action_size, hidden_layer, hidden2_layer, input_channels=4, input_channels2=4, output_channels=16, output_channels2=16, 
                            kernel_size=8, kernel_size2=4, stride=4, stride2=2):
        self.seed = torch.manual_seed(seed)
        self.actor = Actor(self, state_size, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2)
        self.actor_target = Actor(self, state_size, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2)
        self.critic = Critic(self, state_size, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2)
        self.critic_target = Critic(self, state_size, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2)
        self.actor_optimizer = optim.Adam(self.Actor.parameters(), lr=lr, weight_decay=wd)
        self.critic_optimizer = optim.Adam(self.Critic.parameters(), lr=lr, weight_decay=wd)
        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.t_step = 0
        
        
    def act(self, states, eps):
        states = torch.from_numpy(states).float().to(device)
        
        self.Actor.eval()
        with torch.no_grad():
            if random.random() < eps:
            
                action_probs = self.Actor(states).cpu().data.numpy()
                actions = np.argmax(action_probs)


            
            else:
                action_probs = self.Actor(states).cpu().data.numpy()
                actions = random.choice(self.Actor(states).cpu.numpy())

        self.Actor.train()
        
        return actions, action_probs

        
    def step(self, states, actions, rewards, next_states, close):
        self.memory.add(states, actions, rewards, next_states, close)
        
        self.t_step = (self.t_step + 1) % update_every
        
        if self.t_step == 0:
            if len(self.memory) > batch_size:
                experience = self.memory.sample()
                Agent.learn(self, experiences=experience, gamma=0.99)
            
            
    def learn(self, experiences: tuple, gamma: float):
        
        states,actions,rewards,next_states,close = experiences
        
        #--------------------------critic update ----------------------------------------#
        actor_target = self.actor_target(next_states)
        Q_target = self.Critic_target(next_states, actor_target)
        critic_local = self.Critic(states, actions)
        
        critic_target = rewards + (gamma * Q_target * (1-close))
        
        critic_loss = F.mse_loss(critic_local, critic_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        #--------------------------actor update -----------------------------------------#
        actor_local = self.Actor(states)
        actor_loss = -self.Critic(states, actor_local).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        
    def soft_update(self, actor_local, actor_target, critic_local, critic_target):
        for local, target in zip(actor_local, actor_target):
            actor_target.data.copy_(tau*local + (1-tau)*target)
            
        for local, target in zip(critic_local, critic_target):
            critic_target.data.copy_(tau*local + (1-tau)*target)