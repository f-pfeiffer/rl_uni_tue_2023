import numpy as np
import torch
import os
import memory as mem   
from feedforward import Feedforward
from utils import running_mean, DiscreteActionWrapper

class QFunction(Feedforward):
    def __init__(self, observation_dim, action_dim, hidden_sizes=[100,100], 
                learning_rate = 0.0002, dueling=False, clipped=False):
        super().__init__(input_size=observation_dim, hidden_sizes=hidden_sizes, 
                        output_size=action_dim, dueling=dueling)
        self.optimizer=torch.optim.Adam(self.parameters(), 
                                        lr=learning_rate, 
                                        eps=0.000001)
        self.loss = torch.nn.SmoothL1Loss() # MSELoss()
        self.clipped = clipped
    
    def fit(self, observations, actions, targets):
        self.train() # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass
        acts = torch.from_numpy(actions)
        pred = self.Q_value(torch.from_numpy(observations).float(), acts)
        # Compute Loss
        loss = self.loss(pred, torch.from_numpy(targets).float())
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def Q_value(self, observations, actions):
        return self.forward(observations).gather(1, actions[:,None])        
    
    def maxQ(self, observations):
        return np.max(self.predict(observations), axis=-1, keepdims=True)
        
    def greedyAction(self, observations):
        return np.argmax(self.predict(observations), axis=-1)


class DQNAgent(object):
    """
    Agent implementing Q-learning with NN function approximation.    
    """
    def __init__(self, observation_space, action_space, **userconfig):
        
        self._observation_space = observation_space
        self._action_space = action_space
        self._action_n = action_space.n 
        self._config = {
            "eps": 1,            # Epsilon in epsilon greedy policies   
            "eps_min": 0.02,
            "eps_decay": 0.95,                     
            "discount": 0.95,
            "buffer_size": int(1e6),
            "batch_size": 128,
            "learning_rate": 0.0002,
            "update_target_every": 20,
            "use_target_net":True,
            "dueling":False,
            "clipped":False,
            "identifier":"unnamed",
        }
        self._config.update(userconfig)        
        self._eps = self._config['eps']
        self._eps_min = self._config['eps_min']
        self._eps_decay = self._config['eps_decay']
        self._dueling = self._config['dueling']
        self._clipped = self._config['clipped']
        if self._clipped and not self._config['use_target_net']:
            print("Clipping only possible for Double DQN!")
            self._clipped = False
        self.identifier = self._config["identifier"]
        
        self.buffer = mem.Memory(max_size=self._config["buffer_size"])
                
        # Q Network
        self.Q = QFunction(observation_dim=self._observation_space.shape[0], 
                           action_dim=self._action_n,
                           learning_rate = self._config["learning_rate"],
                           dueling = self._dueling,
                           clipped = self._clipped)
        # Q Network
        self.Q_target = QFunction(observation_dim=self._observation_space.shape[0], 
                                  action_dim=self._action_n,
                                  learning_rate = 0,
                                  dueling = self._dueling,
                                  clipped = self._clipped)
        self._update_target_net()
        self.train_iter = 0
            
    def _update_target_net(self):      
        self.Q_target.load_state_dict(self.Q.state_dict())

    def act(self, observation, eps=None):
        if eps is None:
            eps = self._eps
        # epsilon greedy
        if np.random.random() > eps:
            action = self.Q.greedyAction(observation)
        else: 
            action = self._action_space.sample()        
        return action
    
    def store_transition(self, transition):
        self.buffer.add_transition(transition)
            
    def train(self, iter_fit=32):
        losses = []
        self.train_iter+=1
        if self._config["use_target_net"] and self.train_iter % self._config["update_target_every"] == 0:
            self._update_target_net()                
        for i in range(iter_fit):

            # sample from the replay buffer
            data=self.buffer.sample(batch=self._config['batch_size'])
            s = np.stack(data[:,0]) # s_t
            a = np.stack(data[:,1]) # a_t
            rew = np.stack(data[:,2])[:,None] # rew  (batchsize,1)
            s_prime = np.stack(data[:,3]) # s_t+1
            done = np.stack(data[:,4])[:,None] # done signal  (batchsize,1)
            
            if self._config["use_target_net"]:
                v_prime = self.Q_target.maxQ(s_prime)
            else:
                v_prime = self.Q.maxQ(s_prime)

            # target
            """ Start of code contribution (Clipping for TD target)"""
            if self._clipped:
                a_prime_online = torch.from_numpy(self.Q.greedyAction(s_prime)) #select action with online network
                a_prime_target = torch.from_numpy(self.Q.greedyAction(s_prime)) #select action with target network
                v_prime_online = self.Q.Q_value(torch.from_numpy(s_prime), a_prime_target) #get q_vals of online network for action by target network
                v_prime_target = self.Q_target.Q_value(torch.from_numpy(s_prime), a_prime_online) #get q_vals of target network for action by online network
                v_prime = torch.min(v_prime_online, v_prime_target).detach().numpy() #use smaller v_prime for td_target
            """ Start of code contribution (Clipping for TD target)"""

            gamma=self._config['discount']                                           
            td_target = rew + gamma * (1.0-done) * v_prime
            
            # optimize the lsq objective
            fit_loss = self.Q.fit(s, a, td_target)
            
            losses.append(fit_loss)
                
        return losses
    
    """ Start of code contribution (epsilon decay + some utils)"""
    def update_eps(self, step):
        decayed_eps = self._eps * (self._eps_decay**step)
        self._eps = max(decayed_eps, self._eps_min)

    def save_agent(self, path, step):
        p = path +'/' + self.identifier + str(step) + '/'
        if not os.path.exists(p):
            os.makedirs(p)
        torch.save(self.Q.state_dict(), p + f'online.pth')
        torch.save(self.Q_target.state_dict(), p + f'target.pth')

    def load_agent(self, path, step):
        p = path #+ str(step).zfill(8) + '/'
        self.Q.load_state_dict(torch.load(p + f'online.pth'))
        self.Q_target.load_state_dict(torch.load(p + f'target.pth'))
    """ End of code contribution (epsilon decay + some utils)"""