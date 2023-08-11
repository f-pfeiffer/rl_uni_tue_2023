import numpy as np
from torch import nn
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch
import os
from utils.replay_buffer import ReplayBuffer


class Critic(nn.Module):
    def __init__(self, lr_critic, input_dim, n_actions):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim + n_actions, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q = nn.Linear(256, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_critic)
        #self.lr_scheduler = StepLR(self.optimizer, step_size=50_000, gamma=0.9)
 
    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        return self.q(x)

class Actor(nn.Module):
    def __init__(self, lr_actor, input_dim, n_actions,):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, n_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_actor)
        #self.lr_scheduler = StepLR(self.optimizer, step_size=50_000, gamma=0.9)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.mu(x))
    
class TD3Agent():
    def __init__(self, lr_critic, lr_actor, gamma, input_dim, tau, n_actions, max_buffer_size, batch_size, update_actor_interval, explore_n_times, noise, device):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_actor_interval = update_actor_interval
        self.explore_n_times = explore_n_times
        self.n_actions = n_actions
        self.noise = noise
        self.step_counter_train = 0
        self.step_counter_act = 0
        self.device = device
        self.actor = Actor(lr_actor=lr_actor, input_dim=input_dim, n_actions=n_actions).to(device)
        self.critic_1 = Critic(lr_critic=lr_critic, input_dim=input_dim, n_actions=n_actions).to(device)
        self.critic_2 = Critic(lr_critic=lr_critic, input_dim=input_dim, n_actions=n_actions).to(device)
        self.target_actor = Actor(lr_actor=lr_actor, input_dim=input_dim, n_actions=n_actions).to(device)
        self.target_critic_1 = Critic(lr_critic=lr_critic, input_dim=input_dim, n_actions=n_actions).to(device)
        self.target_critic_2 = Critic(lr_critic=lr_critic, input_dim=input_dim, n_actions=n_actions).to(device)
        self.replay_buffer = ReplayBuffer(max_size=max_buffer_size, input_dim=input_dim, n_actions=n_actions, device=device)
        self.update_network_parameters(tau=1)
        self.identifier = 'TD3'


    def act(self, observation):
        if self.step_counter_act < self.explore_n_times:  # do random actions to explore the environment
            actor_pred = torch.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))).to(self.device)
        else:
            observation = torch.tensor(observation, dtype=torch.float).to(self.device)
            actor_pred = self.actor.forward(observation).to(self.device)
        action = actor_pred + torch.tensor(np.random.normal(scale=self.noise), dtype=torch.float).to(self.device)  # add noise
        self.step_counter_act += 1
        actions = action.cpu().detach().numpy()
        return actions
    
    def remote_act(self, observation):
        observation = torch.tensor(observation, dtype=torch.float).to(self.device)
        action = self.actor.forward(observation).to(self.device)
        actions = action.cpu().detach().numpy()
        return actions
    
    def store_transition(self, state, action, reward, state_new, done):
        self.replay_buffer.store_transition(state, action, reward, state_new, done)

    def train(self):
        if self.replay_buffer.counter < self.batch_size:
            return
        
        state, action, reward, new_state, done = self.replay_buffer.sample_transitions(self.batch_size)

        target_actions = self.target_actor.forward(new_state)
        target_actions = target_actions + torch.clamp(torch.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)  # clamp the gaussian noise to not distort the action too much
        target_actions = torch.clamp(target_actions, -1.0, 1.0)  # clamp the action to the action space

        Q1_prime = self.target_critic_1.forward(new_state, target_actions)
        Q2_prime = self.target_critic_2.forward(new_state, target_actions)
        Q1_prime[done] = 0.0
        Q2_prime[done] = 0.0
        Q1_prime = Q1_prime.flatten()
        Q2_prime = Q2_prime.flatten()

        Q1 = self.critic_1.forward(state, action)
        Q2 = self.critic_2.forward(state, action)

        y1 = reward + self.gamma * torch.min(Q1_prime, Q2_prime)
        y1 = y1.view(self.batch_size, 1)

        # optimize the critic
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        loss_critic_1 = F.mse_loss(y1, Q1)
        loss_critic_2 = F.mse_loss(y1, Q2)
        critic_loss = loss_critic_1 + loss_critic_2
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.step_counter_train += 1

        # optimize the actor every other step
        if self.step_counte_train % self.update_actor_interval == 0:
            self.actor.optimizer.zero_grad()
            loss_actor = -torch.mean(self.critic_1.forward(state, self.actor.forward(state)))
            loss_actor.backward()
            self.actor.optimizer.step()
            self.update_network_parameters(self.tau)

        '''
        self.actor.lr_scheduler.step()
        self.critic_1.lr_scheduler.step()
        self.critic_2.lr_scheduler.step()
        '''

    def update_network_parameters(self, tau):
        actor_state_dict = dict(self.actor.named_parameters())
        critic_1_state_dict = dict(self.critic_1.named_parameters())
        critic_2_state_dict = dict(self.critic_2.named_parameters())
        target_actor_state_dict = dict(self.target_actor.named_parameters())
        target_critic_1_state_dict = dict(self.target_critic_1.named_parameters())
        target_critic_2_state_dict = dict(self.target_critic_2.named_parameters())

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau*critic_1_state_dict[name].clone() + (1-tau)*target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau*critic_2_state_dict[name].clone() + (1-tau)*target_critic_2_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_agent(self, path, step):
        p = path + str(step).zfill(8) + '/'
        if not os.path.exists(p):
            os.makedirs(p)
        torch.save(self.actor.state_dict(), p + f'actor.pth')
        torch.save(self.critic_1.state_dict(), p + f'critic_1.pth')
        torch.save(self.critic_2.state_dict(), p + f'critic_2.pth')
        torch.save(self.target_actor.state_dict(), p + f'target_actor.pth')
        torch.save(self.target_critic_1.state_dict(), p + f'target_critic_1.pth')
        torch.save(self.target_critic_2.state_dict(), p + f'target_critic_2.pth')

    def load_agent(self, path, step):
        p = path + str(step).zfill(8) + '/'
        print(f'Loading agent from {p} ...')
        self.actor.load_state_dict(torch.load(p + f'actor.pth', map_location=self.device))
        self.critic_1.load_state_dict(torch.load(p + f'critic_1.pth', map_location=self.device))
        self.critic_2.load_state_dict(torch.load(p + f'critic_2.pth', map_location=self.device))
        self.target_actor.load_state_dict(torch.load(p + f'target_actor.pth', map_location=self.device))
        self.target_critic_1.load_state_dict(torch.load(p + f'target_critic_1.pth', map_location=self.device))
        self.target_critic_2.load_state_dict(torch.load(p + f'target_critic_2.pth', map_location=self.device))