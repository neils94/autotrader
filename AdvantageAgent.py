from Model import Actor, Critic, DQN
import random
import torch
import torch.nn.functional as F
from torch import optim
import ReplayBuffer
import numpy as np
from numpy import random
import torch.nn
from ReplayBuffer import ReplayBuffer
import torch.multiprocessing
from Environment import environment

alpha = 0.99
beta = 0.01
eps= 0.9
gamma = 0.99
lamb = 1e-2
tau = 1e-3


lr = 1e-4
wd = 1e-4
buffer_size = int(1e5)
batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

update_every = 16

class AC_agent():
    """
    """
    
    def __init__(self, batch_size, buffer_size, state, linear_input, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2):
        self.Actor = Actor(state, linear_input, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2)
        self.Critic = Critic(state, linear_input, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2)
        self.Critic_target = Critic(state,linear_input,action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2)
        self.Actor_target = Actor(state,linear_input, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2)
        self.Actor_optimizer = optim.Adam(self.Actor.parameters(), lr=lr, eps=eps, weight_decay=wd)
        self.Critic_optimizer = optim.Adam(self.Critic.parameters(), lr=lr, eps=eps, weight_decay=wd)
        self.DQN = DQN(state, linear_input, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2)
        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.t_step = 0
        self.env = environment()
    
    def act(self, state):
        state = self.env.preprocess(state)
        self.Actor.eval()
        with torch.no_grad():
            action_probs, action = torch.max(self.Actor(state), dim=-1, keepdim=False, out=None)
        self.Actor.train()
        return action_probs, action


    def step(self, state, action, action_probs, reward, next_state, close):
        self.memory.add(state, action, action_probs, reward, next_state, close)

        t_step = self.t_step % update_every

        if t_step == 0:
            if len(self.memory) > batch_size:

                experience = self.memory.sample()
                self.learn(experience, gamma)
            if len(self.memory) % 500:
                self.slow_updates(self.Actor, self.Actor_target, self.Critic, self.Critic_target)


    def learn(self, experiences, gamma):
        state, action, action_probs, reward, next_state, close = experiences
        #------------------------critic loss----------------------#
        _,  actor_target = self.act(state)
        Q_target = self.Critic_target(next_state, actor_target)
        state_value = self.Critic(state, action)

        Q_value = reward + (gamma * Q_target * close)

        critic_loss = F.mse_loss(state_value, Q_value)

        self.Critic_optimizer.zero_grad()
        critic_loss.backward()
        self.Critic_optimizer.step()

        #--------------------actor loss--------------------------#
        actor_current = torch.log(self.Actor(state))
        actor_target = self.Critic(next_state, actor_current)
        advantage = Q_value - state_value
        entropy = -np.sum(np.mean(Q_value) * np.log(Q_value))
        actor_loss = advantage * -actor_target + entropy + 0.001

        self.Actor_optimizer.zero_grad()
        actor_loss.backward()
        self.Actor_optimizer.step()


    def slow_updates(self,actor_current, actor_target, critic_current, critic_target):
        for current, target in zip(actor_current, actor_target):
            self.Actor.data.copy_(tau*current + (1-tau)*target)
             
        for current, target in zip(critic_current, critic_target):
            self.Critic.data.copy_(tau*current + (1-tau)*target)
