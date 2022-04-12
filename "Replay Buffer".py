"Replay Buffer"

class ReplayBuffer():
    def __init__(self, buffer_size, batch_size):
        self.experience = namedtuple(typename = "experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.memory = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        
    def add(self, states, actions, rewards, next_states, dones):
        exp = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(exp)
        
        
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack(exp.states for exp in experiences if exp is not None)).float().to(device)
        actions = torch.from_numpy(np.vstack(exp.actions for exp in experiences if exp is not None)).float().to(device)
        rewards = torch.from_numpy(np.vstack(exp.rewards for exp in experiences if exp is not None)).float().to(device)
        next_states = torch.from_numpy(np.vstack(exp.next_states for exp in experiences if exp is not None)).float().to(device)
        dones = torch.from_numpy(np.vstack(exp.dones for exp in experiences if exp is not None).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)