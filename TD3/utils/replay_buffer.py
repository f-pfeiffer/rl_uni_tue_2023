import numpy as np
import torch


class ReplayBuffer():
    """
    A Replay Buffer class for storing and sampling trainsitions in the form of (s, a, r, s', done).
    This class handles torch tensors.
    """

    def __init__(self, max_size, input_dim, n_actions, device):
        """
        Parameters
        ----------
        max_size : int
            Maximum number of trainsitions to store in the buffer.
        input_shape : tuple
            Shape of the state.
        n_actions : int
            Number of actions.
        device : torch.device
            Device on which the tensors are returned. (Always stored on CPU to avoid GPU memory usage)
        """

        self.size = max_size
        self.counter = 0  # index of last stored experience
        self.device = device
        self.device_cpu = torch.device('cpu')
        self.state_buffer = torch.zeros((self.size, input_dim), dtype=torch.float32).to(self.device_cpu)
        self.state_new_buffer = torch.zeros((self.size, input_dim), dtype=torch.float32).to(self.device_cpu)
        self.action_buffer = torch.zeros((self.size, n_actions), dtype=torch.float32).to(self.device_cpu)
        self.reward_buffer = torch.zeros(self.size, dtype=torch.float32).to(self.device_cpu)
        self.terminal_buffer = torch.zeros(self.size, dtype=torch.long).to(self.device_cpu)

    def store_transition(self, state, action, reward, state_new, done):
        """
        Store a transition in the replay buffer. Takes numpy arrays and torch tensors as input.
        """
        index = self.counter % self.size  # index of first available memory (oldest)
        self.state_buffer[index] = state = torch.as_tensor(state, dtype=torch.float32).detach().to(self.device_cpu)
        self.state_new_buffer[index] = torch.as_tensor(state_new, dtype=torch.float32).detach().to(self.device_cpu)
        self.reward_buffer[index] = torch.as_tensor(reward, dtype=torch.float32).detach().to(self.device_cpu)
        self.action_buffer[index] = torch.as_tensor(action, dtype=torch.float32).detach().to(self.device_cpu)
        self.terminal_buffer[index] = torch.as_tensor(done, dtype=torch.int).detach().to(self.device_cpu)
        self.counter += 1

    def sample_transitions(self, batch_size):
        max_idx = min(self.counter, self.size)  # don't sample more than available memory
        batch = np.random.choice(max_idx, batch_size, replace=False)  # sample without replacement
        states = self.state_buffer[batch]
        states_new = self.state_new_buffer[batch]
        rewards = self.reward_buffer[batch]
        actions = self.action_buffer[batch]
        terminals = self.terminal_buffer[batch]
        return states.to(self.device), actions.to(self.device), rewards.to(self.device), states_new.to(self.device), terminals.to(self.device)
