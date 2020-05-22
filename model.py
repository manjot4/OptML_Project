# Importing Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# num_units = 512
# # Network/Model
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(784, num_units)
#         self.bn1= nn.BatchNorm1d(num_features=num_units)
#         self.fc2 = nn.Linear(num_units, num_units)
#         self.bn2 = nn.BatchNorm1d(num_features=num_units)
#         self.fc3 = nn.Linear(num_units, num_units)
#         self.bn3 = nn.BatchNorm1d(num_features=num_units)
#         self.fc4 = nn.Linear(num_units, num_units)
#         self.bn4 = nn.BatchNorm1d(num_features=num_units)
#         self.fc5 = nn.Linear(num_units, 10)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(self.bn1(x))
#         x = self.fc2(x)
#         x = F.relu(self.bn2(x))
#         x = self.fc3(x)
#         x = F.relu(self.bn3(x))
#         x = self.fc4(x)
#         x = F.relu(self.bn4(x))
#         x = self.fc5(x)
#         x = F.relu(x)
#         output = F.log_softmax(x, dim=1)
#         return output


# Network/Model
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
        




# # Network/Model
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout2d(0.25)
#         self.dropout2 = nn.Dropout2d(0.5)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output
    
    

