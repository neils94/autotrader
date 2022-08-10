from collections import namedtuple, deque
import random
import tensorflow as tf


lr = 2e-4
wd = 1e-4
lamb = 1e-2
gamma = 0.99
tau = 1e-3
buffer_size = int(1e5)
batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0
update_every = 16


class Agent():

    
    def __init__(self, batch_size, buffer_size, state_size, action_size, hidden_layer, hidden2_layer):
        self.seed = torch.manual_seed(seed)
        self.Actor = Actor(state_size, action_size, hidden_layer, hidden2_layer, seed) 
        self.Actor_target = Actor(state_size, action_size, hidden_layer, hidden2_layer, seed) 
        self.Critic = Critic(state_size, action_size, hidden_layer, hidden2_layer, seed)
        self.Critic_target = Critic(state_size, action_size, hidden_layer, hidden2_layer, seed)
        self.actor_optimizer = optim.Adam(self.Actor.parameters(), lr=lr, weight_decay=wd)
        self.critic_optimizer = optim.Adam(self.Critic.parameters(), lr=lr, weight_decay=wd)
        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.t_step = 0
        
        
    def act(self, states):
        states = torch.from_numpy(states).float().to(device)
        
        self.Actor.eval()
        with torch.no_grad():
            
            actions = self.Actor(states).cpu().data.numpy()
            
        self.Actor.train()
        
        return actions

        
    def step(self, states, actions, rewards, next_states, trade):
        self.memory.add(states, actions, rewards, next_states, trade)
        
        self.t_step = (self.t_step + 1) % update_every
        
        if self.t_step == 0:
            if len(self.memory) > batch_size:
                experience = self.memory.sample()
                Agent.learn(self, experiences=experience, gamma=0.99)
            
            
    def learn(self, experiences, gamma):
        
        states,actions,rewards,next_states,dones = experiences
        
        #--------------------------critic update ----------------------------------------#
        actor_target = self.Actor_target(next_states)
        Q_target = self.Critic_target(next_states, actor_target)
        critic_local = self.Critic(states, actions)
        
        critic_target = rewards + (gamma * Q_target * (1-dones))
        
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
            actor_target.data.copy_(tau*actor_local + (1-tau)*actor_target)
            
        for local, target in zip(critic_local, critic_target):
            critic_target.data.copy_(tau*critic_local + (1-tau)*critic_target)
