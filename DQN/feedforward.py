import torch
import numpy as np

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dueling=False):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes  = hidden_sizes
        self.output_size  = output_size
        self.dueling = dueling
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [ torch.nn.Tanh() for l in  self.layers ]

        """ Start of code contribution (Dueling modification) """
        if self.dueling:
            self.advantage_head = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_sizes[-1], 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, self.output_size)
                )
            self.value_head = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_sizes[-1], 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1)
                )
        else:
            self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)
        """ End of code contribution (Dueling modification)"""

    def forward(self, x):
        for layer,activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))

        """ Start of code contribution (Dueling modification) """
        if self.dueling:
            advantage = self.advantage_head(x) # advantage value per action
            value = self.value_head(x) # state value
            q_value = value - (advantage - advantage.mean())
        else:
            q_value = self.readout(x)
        """ End of code contribution (Dueling modification) """

        return q_value

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32))).numpy()
