# Importing Libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import argparse, matplotlib.pyplot as plt


def dataloaders(batch_size, use_cuda):
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    
    return train_loader, test_loader


#### path:= "./your_directory/checkpoint_name.tar",
def save_checkpoint(state, checkpoint_path):
    """
    state: checkpoint we want to save
    checkpoint_path: path to save checkpoint
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    return 
 

def load_ckp(checkpoint_path, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_path)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def train(model, device, train_loader, optimizer, epoch, batch_size):
    model.train()
    running_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # print (data.size())
        data = data.view(-1, 784)
        # print (data.size())
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() 
    train_loss = running_loss/len(train_loader)
    return train_loss

#         if batch_idx % log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))



# we are getting the loss for one batch....



def test(model, device, test_loader, batch_size):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 28*28)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='mean').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss
    
    
def show_losses(train_losses, test_losses):
    fig = plt.figure()
    plt.plot(train_losses, color='blue')
    plt.plot(test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('Loss')
    fig.show()
    

def show_sharpness_norm(l_sharpness, l_weight_norms, sigmas):
    # for i in range(len(sigmas)):
    #     sigmas[i] = str(sigmas[i])
    fig = plt.figure()
    plt.plot(sigmas, l_sharpness, label = "sharpness",  marker='o')
    plt.plot(sigmas, l_weight_norms, label = "norms",  marker='o')
    # plt.plot(l_weight_norms, l_sharpness, label = "S", marker='o', color='b')
    plt.grid(True, linestyle='-.')
    plt.legend() #loc='lower left'
    plt.title("Norm and Sharpness")
    plt.xlabel("Sigma")
    plt.ylabel("Measure")
    plt.show()	        
###############################
    
# Rough Work    
    
# #pruning 
#     total = 0
#     for m in model.modules():
#         if isinstance(m, nn.Conv2d):
#             total += m.weight.data.numel()
#     conv_weights = torch.zeros(total)
#     index = 0
#     for m in model.modules():
#         if isinstance(m, nn.Conv2d):
#             size = m.weight.data.numel()
#             conv_weights[index:(index+size)] = m.weight.data.view(-1).abs().clone()
#             index += size 

# d = (torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0])))
# # print (d.sample((5,)).squeeze())

# a = torch.zeros([2,2])
# b = d.sample((2,2))
# print (b)
# a = b.squeeze()
# print (a)
