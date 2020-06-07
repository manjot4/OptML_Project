# Importing Libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import argparse, matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from measures import *


def dataloaders(batch_size, use_cuda, seed):
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=batch_size, shuffle=True, **kwargs)
  
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                          transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))

    val_dataset = datasets.MNIST('../data', train=True, download=True,
                          transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
    valid_size = 0.1  
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    random_seed = seed
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    # print ("chckp1::", len(valid_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # print ("chckp1::", len(valid_sampler))

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                    batch_size=batch_size, sampler=train_sampler, **kwargs)

    val_loader = torch.utils.data.DataLoader(val_dataset, 
                    batch_size=batch_size, sampler=valid_sampler, **kwargs)                
    # print ("chckp1::", len(val_loader))    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    
    return train_loader, val_loader, test_loader, len(train_sampler), len(valid_sampler)


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
        data = data.view(-1, 784)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() 
    train_loss = running_loss/len(train_loader)
    return train_loss


def train_get_grad(model, device, train_loader, optimizer, epoch, batch_size):
    model.train()
    grad_norms = []
    grad_avg = torch.zeros([1861632])
    running_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(-1, 784)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        # optimizer.step()
        # running_loss += loss.item() 

        gradients = get_gradients(model)
        # print (gradients.size())
        # print ("ok")
        norm = torch.norm(gradients)
        grad_norms.append(norm*norm)
        # mean_grad = torch.mean(gradients,0)
        # grad_avg.append(mean_grad)
        # grad_avg.append(torch.mean(gradients).item())
        # grad_avg.append(gradients)
        grad_avg += gradients 
        del gradients
    train_loss = running_loss/len(train_loader)
    grad_avg = grad_avg / len(train_loader)
    return train_loss, grad_norms, grad_avg

#         if batch_idx % log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))



def train_kfac_get_grad(model, device, train_loader, optimizer, epoch, batch_size):
    model.train()
    grad_norms = []
    grad_avg = torch.zeros([1861632])
    # grad_avg = []
    running_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(-1, 784)
        optimizer.zero_grad()
        output = model(data) 
        loss = F.nll_loss(output, target)
        # if optim_name in ['kfac', 'ekfac'] and optimizer.steps % optimizer.TCov == 0:
        if optimizer.steps % optimizer.TCov == 0:
            # compute true fisher
            optimizer.acc_stats = True
            with torch.no_grad():
                sampled_y = torch.multinomial(torch.nn.functional.softmax(output.cpu().data, dim=1),1).squeeze().to(device)
            loss_sample = F.nll_loss(output, sampled_y)
            loss_sample.backward(retain_graph=True)
            optimizer.acc_stats = False
            optimizer.zero_grad()  # clear the gradient for computing true-fisher.
        loss.backward()
        optimizer.step()

        gradients = get_gradients(model)
        norm = torch.norm(gradients)
        grad_norms.append(norm*norm)#.tolist())
        # mean_grad = torch.mean(gradients,0)
        # grad_avg.append(mean_grad)
        # grad_avg.append(torch.mean(gradients).item())
        grad_avg += gradients 
        del gradients

        running_loss += loss.item() 
    train_loss = running_loss/len(train_loader)
    grad_avg = grad_avg / len(train_loader)
    return train_loss, grad_norms, grad_avg


def train_kfac(model, device, train_loader, optimizer, epoch, batch_size):
    model.train()
    running_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(-1, 784)
        optimizer.zero_grad()
        output = model(data) 
        loss = F.nll_loss(output, target)
        # if optim_name in ['kfac', 'ekfac'] and optimizer.steps % optimizer.TCov == 0:
        if optimizer.steps % optimizer.TCov == 0:
            # compute true fisher
            optimizer.acc_stats = True
            with torch.no_grad():
                sampled_y = torch.multinomial(torch.nn.functional.softmax(output.cpu().data, dim=1),1).squeeze().to(device)
            loss_sample = F.nll_loss(output, sampled_y)
            loss_sample.backward(retain_graph=True)
            optimizer.acc_stats = False
            optimizer.zero_grad()  # clear the gradient for computing true-fisher.
        loss.backward()
        optimizer.step()
        running_loss += loss.item() 
    train_loss = running_loss/len(train_loader)
    return train_loss




def test(model, device, test_loader, batch_size):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 28*28)
            output = model(data)
            test_loss += F.nll_loss(output, target).item()  
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    # test_loss /= len(test_loader.dataset)
    test_loss = test_loss / len(test_loader) 
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss


def test_2(model, device, test_loader, batch_size, num_samples):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 28*28)
            output = model(data)
            test_loss += F.nll_loss(output, target).item()  
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = test_loss / len(test_loader) 
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, num_samples,
        100. * correct / num_samples))
    return test_loss

 
    
def show_losses(train_losses, test_losses):
    fig = plt.figure()
    plt.plot(train_losses, color='blue')
    plt.plot(test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of epochs')
    plt.ylabel('Loss')
    fig.show()
    

# def show_sharpness_norm(l_sharpness, l_weight_norms, sigmas):
#     # for i in range(len(sigmas)):
#     #     sigmas[i] = str(sigmas[i])
#     fig = plt.figure()
#     plt.plot(sigmas, l_sharpness, label = "sharpness",  marker='o')
#     plt.plot(sigmas, l_weight_norms, label = "norms",  marker='o')
#     # plt.plot(l_weight_norms, l_sharpness, label = "S", marker='o', color='b')
#     plt.grid(True, linestyle='-.')
#     plt.legend() #loc='lower left'
#     plt.title("Norm and Sharpness")
#     plt.xlabel("Sigma")
#     plt.ylabel("Measure")
#     plt.show()            
# ###############################