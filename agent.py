# Import Packages for Agent Class
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random

from model import QNetwork


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
PRB_A = 0.6             # importance sampling parameter
PRB_B = 0.4             # prioritization parameter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class DQNAgent():
    """Agent class to interact with enviroment."""
    
    def __init__(self, state_size, action_size, seed, use_dueling=False, use_double=False):
        """Initialize an Agent object."""
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.use_dueling = use_dueling
        self.use_double = use_double
        
        # Q-Networks
        self.qnetwork_local = QNetwork(state_size, action_size, seed, use_dueling=use_dueling).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, use_dueling=use_dueling).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
        
    def act(self, state, eps=0.):
        """Return an action based on the given state."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # Epsilon greedy selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self, experiences, gamma):
        """Update value parameters using given sampled batch of experiences."""
        states, actions, rewards, next_states, dones = experiences
        
        # Compute and minimize the loss
        # Get max predicted Q values (for next states) from target model
        if self.use_double:
            indices = torch.argmax(self.qnetwork_local(next_states).detach(),1)
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1,indices.unsqueeze(1))
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update the target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        
    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)        
        
class DQNPRBAgent(DQNAgent):
    """DQN for priotised Replay Buffer."""
    def __init__(self, state_size, action_size, seed, a=PRB_A, max_t=1000, init_b=PRB_B, use_dueling=False, use_double=False):
        """Initialise DQN Agent."""
        super(DQNPRBAgent, self).__init__(state_size, action_size, seed)
        self.memory = PrioritisedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, a)
        self.a = a
        self.init_b = init_b
        self.max_t = max_t
        self.t_step = 0
        self.use_dueling = use_dueling
        self.use_double = use_double
        
    def get_beta(self, t):
        """Return B, based on timestep. B increases over time step."""
        fraction = min(float(t) / self.max_t, 1.0)
        current_beta = self.init_b + fraction * (1.0 - self.init_b)
        
        return current_beta
    
    def wighted_mse_loss(self, input_tensor, target_tensor, weights):
        """ Calculate the weighted mse loss."""
        out = (input_tensor - target_tensor) ** 2
        out = out * weights.expand_as(out)
        loss = out.mean(0)
        return loss
    
    def learn(self, experiences, gamma):
        """Update value parameters using given sampled batch of experiences."""
        states, actions, rewards, next_states, dones = experiences
        
        # Compute and minimize the loss
        # Get max predicted Q values (for next states) from target model
        if self.use_double:
            indices = torch.argmax(self.qnetwork_local(next_states).detach(),1)
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1,indices.unsqueeze(1))
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Calculate importance sampling weight
        current_beta = self.get_beta(self.t_step)
        # Get weights
        weights = self.memory.get_weights(current_beta)
        
        td_errors = Q_targets_next - Q_expected
        self.memory.update_priorities(td_errors)
        
        # Calculate loss
        loss = self.wighted_mse_loss(Q_expected, Q_targets_next, weights)
        # Clear the gradients
        self.optimizer.zero_grad()
        # Gradient decent
        loss.backward()
        self.optimizer.step()
        
        
class ReplayBuffer():
    """Replay memory to store experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object."""
        self.action_size = action_size
        self.memory = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to the memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        """Random sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the size of the memory."""
        return len(self.memory)
    
    
class PrioritisedReplayBuffer():
    """Replay memory to store experience tuples with prioritisation"""
    
    def __init__(self, action_size, buffer_size, batch_size, seed, a):
        """Initialize replay prioritised replay buffer."""
        self.action_size = action_size
        self.memory = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.a = a
        self.priorities = deque(maxlen = buffer_size)
        self.buffer_size = buffer_size
        self.sum_priorities = 0.0
        self.eps = 1e-6
        self.indexes = []
        self.max_priority = 1.0 ** self.a
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to the memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        # Remove the priority value removed from the least recent experience from sum
        if len(self.priorities) >= self.buffer_size:
            self.sum_priorities -= self.priorities[0]
        self.priorities.append(self.max_priority)
        # Add the priority value added to the most recent experience from sum
        self.sum_priorities += self.priorities[-1]
        
    def sample(self):
        """Random sample a batch of experiences from memory."""
        m_len = len(self.memory)
        if self.sum_priorities:
            na_probs = np.array(self.priorities)/self.sum_priorities
        self.c_index = np.random.choice(m_len, size=min(m_len, self.batch_size), p=na_probs)
        experiences = [self.memory[i] for i in self.c_index]
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def calculate_w(self, f_priority, current_b, max_weight, m_len):
        f_wi = (m_len * f_priority / self.sum_priorities)
        return (f_wi ** -current_b) / max_weight
    
    def get_weights(self, current_b):
        """Return the importance sampling weights of current batch"""
        m_len = len(self.memory)
        max_weight = (m_len * min(self.priorities) / self.sum_priorities)
        max_weight = max_weight ** -current_b
        
        weights = [self.calculate_w(self.priorities[i], current_b, max_weight, m_len) for i in self.c_index]
        
        return torch.tensor(weights, device=device, dtype=torch.float).reshape(-1, 1)
    
    def update_priorities(self, td_errors):
        """Update priorities."""
        for index, td_error in zip(self.c_index, td_errors):
            td_error = float(td_error)
            self.sum_priorities -= self.priorities[index]
            self.priorities[index] = ((abs(td_error) + self.eps) ** self.a)
            self.sum_priorities += self.priorities[index]
        self.max_priority = max(self.priorities)
        self.c_index = []
        
    def __len__(self):
        """Return the size of the memory."""
        return len(self.memory)      
        
            