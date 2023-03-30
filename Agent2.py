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

class Agent_2():
    """
    """
    
    def __init__(self, batch_size, buffer_size, state, linear_input, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2):
        self.Actor = Actor(state, linear_input, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2)
        self.Actor_target = Actor(state, linear_input, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2)
        self.Actor2 = Actor(state, linear_input, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2)
        self.Actor2_target = Actor(state, linear_input, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2)
        self.Critic = Critic(state, linear_input, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2)
        self.Critic2 = Critic(state, linear_input, action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2)
        self.Critic_target = Critic(state,linear_input,action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2)
        self.Critic2_target = Critic(state,linear_input,action_size, hidden_layer, hidden2_layer, input_channels, input_channels2, output_channels, output_channels2, 
                            kernel_size, kernel_size2, stride, stride2)
        self.Actor_optimizer = optim.Adam(self.Actor.parameters(), lr=lr, eps=eps, weight_decay=wd)
        self.Actor2_optimizer = optim.Adam(self.Actor2.parameters(), lr=lr, eps=eps, weight_decay=wd)
        self.Critic_optimizer = optim.Adam(self.Critic.parameters(), lr=lr, eps=eps, weight_decay=wd)
        self.Critic2_optimizer = optim.Adam(self.Critic2.parameters(), lr=lr, eps=eps, weight_decay=wd)
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

    def action_2(self, next_state):
        self.Actor2.eval()
        with torch.no_grad():
            action_2 = torch.argmax(self.Actor2(next_state).cpu())

        self.Actor2.train()
        return action_2

    def step(self, state, action_1, action_2, action_probs, reward, next_state_1, next_state_2,close):
        self.memory.add(state, action_1, action_2, action_probs, reward, next_state_1, next_state_2, close)

        t_step = self.t_step % update_every
        if t_step == 0:
            
            if len(self.memory) > batch_size:

                experience = self.memory.sample()
                self.learn(experience, gamma)
                if len(self.memory) % 500 == 0:
                    self.slow_updates(self.Actor, self.Actor_target, self.Critic, self.Critic_target)
                    actor1_path = "actor1_state_dict.pt"
                    actor2_path = "actor2_state_dict.pt"
                    critic1_path = "critic1_state_dict.pt"
                    critic2_path = "critic2_state_dict_path.pt"
                    torch.save(self.Actor.state_dict(), actor1_path)
                    torch.save(self.Criic.state_dict(), critic1_path)
                    torch.save(self.Actor2.state_dict(), actor2_path)
                    torch.save(self.Critic2.state_dict(), critic2_path)


    def memory_update(self, state, action_1, action_2, action_probs, reward, next_state_1, next_state_2, close):
        self.memory.add(state, action_1, action_2, action_probs, reward, next_state_1, next_state_2, close)
        torch.save(self.memory, "replayBufferAgent2.pt")

    def backprop_network(self):
        experience = self.memory.sample()
        self.learn(experience, gamma)

        actor_path = "actor2_state_dict.pt"
        critic_path = "critic_state_dict_path.pt"
        torch.save(self.Actor.state_dict(), actor_path)
        torch.save(self.Criic.state_dict(), critic_path)

        if len(self.memory) % 500 == 0:
            self.soft_updates(self.Actor, self.Actor2, self.Actor_target, self.Actor2_target, self.Critic, self.Critic_target)

    def learn(self, experiences, gamma):
        state, action_1, action_2, action_prob, reward, next_state_1, next_state_2, close = experiences
        #------------------------critic 1 loss----------------------#

        actor_target = torch.log(self.Actor_target(state))
        Q_target = self.Critic_target(next_state_1, actor_target)
        state_value = self.Critic(state, action_1)

        Q_value = reward + (gamma * Q_target * close)

        critic_loss = F.mse_loss(state_value, Q_value)

        self.Critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), norm_type=2.0, error_if_nonfinite=True)
        self.Critic_optimizer.step()

        #--------------------actor 1 loss--------------------------#
        actor_current = torch.log(self.Actor(state))
        actor_target = self.Critic(next_state_1, actor_current)
        advantage = Q_value - state_value
        actor_loss = advantage * -actor_target

        self.Actor_optimizer.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), norm_type=2.0, error_if_nonfinite=True)
        self.Actor_optimizer.step()

    #------------------------critic 2 loss----------------------#


        actor2_target = torch.log(self.Actor2_target(next_state_1))
        Q_target = self.Critic2_target(next_state_2, actor2_target)
        state_value2 = self.Critic(next_state_1, action_2)

        Q_value2 = reward + (gamma * Q_target * close)

        critic2_loss = F.mse_loss(state_value2, Q_value2)

        self.Critic2_optimizer.zero_grad()
        critic2_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.Critic2.parameters(), norm_type=2.0, error_if_nonfinite=True)
        self.Critic2_optimizer.step()

    #--------------------actor2 loss--------------------------#
        actor2_current = torch.log(self.Actor2(next_state_1))
        actor2_target = self.Critic(next_state_2, actor2_current)
        advantage = Q_value2 - state_value2
        actor2_loss = advantage * -actor2_target
        

        self.Actor2_optimizer.zero_grad()
        actor2_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.Actor2.parameters(), norm_type=2.0, error_if_nonfinite=True)
        self.Actor2_optimizer.step()

    def soft_updates(self,actor_current, actor2_current, actor_target, actor2_target, critic_current, critic_target, critic2_current, critic2_target):
        for current, target in zip(actor_current, actor_target):
            target.data.copy_(tau*current + (1-tau)*target)
        for current, target in zip(actor2_current, actor2_target):
            target.data.copy_(tau*current + (1-tau)*target)
             
        for current, target in zip(critic_current, critic_target):
            target.data.copy_(tau*current + (1-tau)*target)
        for current, target in zip(critic2_current, critic2_target):
            target.data.copy_(tau*current + (1-tau)*target)
    def script_Models(self):
        sx = self.env.screenshot()
        state = self.env.preprocess(sx)
        actor = self.Actor(state)
        scripted_Actor = torch.jit.script(actor)
