"""This file contains all the utilities such as dataloaders, train function, test function etc."""


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
    """ create train, validation and test dataloader
    Parameters
    ----------
    batch_size: integer
        batch_size used for training
    use_cuda: Bool
        use cuda or not
    seed: integer
        for setting seed
    Returns
    -------
    dataloaders:
        returning train, test and validation dataloader
    """
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  
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



def train(model, device, train_loader, optimizer, epoch, batch_size):
    """ trains the model
    Parameters
    ----------
    model: 
        Neural Network
    device:
        GPU name
    train_loader: iterator
        train dataloader
    optimizer:
        optimizer
    epcoh: integer
        number of epochs to train for 
    batch_size: integer
          mini-batch-size
    Returns
    -------
    train loss: float
        training loss
    """
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
    """ get gradients for every mini-batch(every data example) but on same model(fixed weight parameters)
    Parameters
    ----------
    model: 
        Neural Network
    device:
        GPU name
    train_loader: iterator
        train dataloader
    optimizer:
        optimizer
    epcoh: integer
        number of epochs to train for 
    batch_size: integer
          mini-batch-size
    Returns
    -------
    grad_norms: float
      gradient norm
    grad_avg: float
      gradient average
    """
    model.train()
    grad_norms = []
    grad_avg = torch.zeros([1861632])  #number of weight parameters in the network
    running_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(-1, 784)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
  
        gradients = get_gradients(model)
        norm = torch.norm(gradients)
        grad_norms.append(norm*norm)
        grad_avg += gradients 
        del gradients
    train_loss = running_loss/len(train_loader)
    grad_avg = grad_avg / len(train_loader)
    return train_loss, grad_norms, grad_avg


def test(model, device, test_loader, batch_size):
    """ testing/evaluation
    Parameters
    ----------
    model: 
        Neural Network
    device:
        GPU name
    test_loader: iterator
        test dataloader
    batch_size: integer
          mini-batch-size
    Returns
    -------
    test loss: float
        testing loss
    """
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
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss


def test_2(model, device, test_loader, batch_size, num_samples):
    """ testing/evaluation
    Parameters
    ----------
    model: 
        Neural Network
    device:
        GPU name
    test_loader: iterator
        test dataloader
    batch_size: integer
          mini-batch-size
    num_samples: float
        size of data
    Returns
    -------
    test loss: float
        testing loss
    """
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
    """ plot train-test curve
    Parameters
    ----------
    train_losses: list
        training losses over the number of epochs
    test_losses: list
        testing losses over the number of epochs
    """
    fig = plt.figure()
    plt.plot(train_losses, color='blue')
    plt.plot(test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of epochs')
    plt.ylabel('Loss')
    fig.show()
    