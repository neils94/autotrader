from re import I
from Model import Actor, Critic
import random
import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
from numpy import random
import torch.nn
from ReplayBuffer import ReplayBuffer
import torch.multiprocessing
from scipy import stats
from meta_API_environment import meta_env
import heapq
import wandb

alpha = 0.99
beta = 0.01
eps= 0.9
gamma = 0.99
lamb = 1e-2
tau = 1e-3


lr = 1e-4
wd = 1e-4
buffer_size = int(1e5)
batch_size = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

update_every = 16

class AC_agent():
    """
    """
    
    def __init__(self, batch_size, buffer_size, state, linear_input, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2):
        self.Actor = Actor(state, linear_input, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2)
        self.Actor_target = Actor(state, linear_input, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2)
        self.Critic = Critic(state, linear_input, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2)
        self.Critic_target = Critic(state,linear_input,action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2)
        self.Actor_2 = Critic(state,linear_input, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2)
        self.Actor_optimizer = optim.Adam(self.Actor.parameters(), lr=lr, eps=eps, weight_decay=wd)
        self.Critic_optimizer = optim.Adam(self.Critic.parameters(), lr=lr, eps=eps, weight_decay=wd)
        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.t_step = 0
        self.env = meta_env()
    
    def greedy_action(self, state):
        self.Actor.eval()
        with torch.no_grad():
            action_probs, action = torch.max(self.Actor(state).cpu(), dim=-1, keepdim=False, out=None)
        self.Actor.train()
        return action_probs, action

    def entropy(self, action_prob):
        entropy = stats.entropy(action_prob)

        return entropy

    def entropy_action(self, state, exp = 0.9):
        self.Actor.eval()
        if random.random() > exp:
            
            with torch.no_grad():
                action_probs, action = torch.min(self.Actor(state).cpu(), dim=-1, keepdim=False, out=None)
        else:
            with torch.no_grad():
                action_probs, action = torch.max(self.Actor(state).cpu(), dim=-1, keepdim=False, out=None)
        self.Actor.train()
        return action_probs, action


    def step(self, state, action, action_probs, reward, next_state, close):
        self.memory.add(state, action, action_probs, reward, next_state, close)

        t_step = self.t_step % update_every
        if t_step == 0:
            
            if len(self.memory) > batch_size:

                experience = self.memory.sample()
                self.learn(experience, gamma)
                if len(self.memory) % 500 == 0:
                    self.slow_updates(self.Actor, self.Actor_target, self.Critic, self.Critic_target)

    def memory_update(self, state, action, action_probs, reward, next_state, close):
        self.memory.add(state, action, action_probs, reward, next_state, close)
        torch.save(self.memory, "replayBuffer.pt")

    def backprop_network(self):
        experience = self.memory.sample()
        self.learn(experience, gamma)

        actor_path = "actor_state_dict.pt"
        critic_path = "critic_state_dict_path.pt"
        torch.save(self.Actor.state_dict(), actor_path)
        torch.save(self.Criic.state_dict(), critic_path)

        if len(self.memory) % 500 == 0:
            self.slow_updates(self.Actor, self.Actor_target, self.Critic, self.Critic_target)

    def learn(self, experiences, gamma):
        state, action, action_prob, reward, next_state, close = experiences
        #------------------------critic loss----------------------#
        actor_target = torch.log(self.Actor_target(state))
        Q_target = self.Critic_target(next_state, actor_target)
        state_value = self.Critic(state, action)

        Q_value = reward + (gamma * Q_target * close)

        critic_loss = F.mse_loss(state_value, Q_value)

        self.Critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), norm_type=2.0, error_if_nonfinite=True)
        self.Critic_optimizer.step()

        #--------------------actor loss--------------------------#
        actor_current = torch.log(self.Actor(state))
        actor_target = self.Critic(next_state, actor_current)
        advantage = Q_value - state_value
        actor_loss = advantage * -actor_target

        self.Actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), norm_type=2.0, error_if_nonfinite=True)
        self.Actor_optimizer.step()
        
    def slow_updates(self,actor_current, actor_target, critic_current, critic_target):
        for current, target in zip(actor_current, actor_target):
            target.data.copy_(tau*current + (1-tau)*target)
             
        for current, target in zip(critic_current, critic_target):
            target.data.copy_(tau*current + (1-tau)*target)
    
    def script_Models(self):
        sx = self.env.screenshot()
        state = self.env.preprocess(sx)
        actor = self.Actor(state)
        scripted_Actor = torch.jit.script(actor)
