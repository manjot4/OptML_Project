"""This file computes all the main measures that is, 
retrieves the gradients from the model,
computes variance of gradient score,
perturbs the model with Gaussian Distribution,
compute L2 weight norm,
computes last term of equation 1 in the report.
"""

# Importing Libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import argparse, matplotlib.pyplot as plt



def get_gradients(model):
    """ this method retrieves the gradients from the model
      Parameters
      ----------
      model: 
          3 layer Neural Network
      Returns
      -------
      tensor: 
          all gradients
          
    """
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            total += module.weight.grad.data.numel()
    grads = torch.zeros(total)
    index = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            size = module.weight.grad.data.numel()
            grads[index:(index+size)] = module.weight.grad.data.view(-1).clone()
            index += size 
    return grads



def compute_grad_variance(grad_norms, grad_avg):
    """ Computes variance of gradients.
      Parameters
      ----------
      grad_norms: list
          norms of gradients computed for each mini-batch - || g ||^2
      grad_avg: list
          gradient values averaged over all mini-batches(training iterations) - E[g]
      Returns
      -------
      float
          variance of gradient dcore 
    """  

    """E[|| g ||^2 ]""" 
    ft = sum(grad_norms) / len(grad_norms)
    
    """|| E[g] ||^2"""
    st = torch.norm(grad_avg)

    variance_score = ft - (st*st)
    return variance_score
    

def weight_pertubation(model, mean, std, device):
    """ This method perturbs a given model with gaussian distribtion.
    Parameters
    ----------
    model: 
        Neural Network
    mean: float
        mean of Gaussian Distribution
    std: float
        standard deviation of Gaussian Distribution
    Returns
    -------
    model
        perturbed model
    """

    total = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            size = module.weight.data.size()
            gauss_dist = torch.distributions.normal.Normal(torch.tensor([mean]), torch.tensor([std]))
            v = gauss_dist.sample((size)).squeeze().to(device)
            module.weight.data = module.weight.data + v
    return model
    
def cal_l2_norm(model):
    """ This method computes the L2 weight norm 
    Parameters
    ----------
    model: 
        Neural Network
    Returns
    -------
    norm_value: float
        weight norm
    """

    total = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            total += module.weight.data.numel()
    weights = torch.zeros(total)
    index = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            size = module.weight.data.numel()
            weights[index:(index+size)] = module.weight.data.view(-1).clone()
            index += size 
    l2_norm = torch.norm(weights)
    return (l2_norm * l2_norm)
    
    
def compute_bound(model, train_size, sigma, weight_l2_norm, delta):
    """ This method computes last term(weight norm) of equation 1 in the report.
      Parameters
      ----------
      model: 
          Neural Network
      train_size: integer
            number of training examples
      sigma: float
          standard deviation
      weight_l2_norm: float
          L2 weight norm
      delta: float
      
      Returns
      -------
      norm_value: float
          norm
    """
    ln_term = torch.tensor((2*train_size) / (delta))
    val = (1/train_size) * ( (weight_l2_norm / (2*sigma*sigma)) + (torch.log(ln_term)) )
    last_term = 4 * ((val)**(0.5))  # it is a tensor value
    return last_term.item()  

