"""This file contains the architecture of the model that we use for our experiments

Description of the model:
It is a 3-layer feedforward neural network. The size of hidden layer is 1024 units.
We use ReLU as an activation function.
"""

# Importing Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        output = F.log_softmax(x, dim=1)
        return output