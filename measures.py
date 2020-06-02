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



def compute_grad_variance(grad_norms, grad_avg):  #list of lists consisting of gradient values....
    """E[|| g ||^2 ]""" 
    ft = sum(grad_norms) / len(grad_norms)
    
    """|| E[g] ||^2"""
    # grad_avg = torch.stack(grad_avg)
    # grad_avg = torch.mean(grad_avg,0)
    st = torch.norm(grad_avg)
    
    variance_score = ft - st
    return variance_score
    

def weight_pertubation(model, mean, std, device):
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            size = module.weight.data.size()
            gauss_dist = torch.distributions.normal.Normal(torch.tensor([mean]), torch.tensor([std]))
            v = gauss_dist.sample((size)).squeeze().to(device)
            module.weight.data = module.weight.data + v
    return model
    
def cal_l2_norm(model):
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
    ln_term = torch.tensor((2*train_size) / (delta))
    val = (1/train_size) * ( (weight_l2_norm / (2*sigma*sigma)) + (torch.log(ln_term)) )
    last_term = 4 * ((val)**(0.5))  # it is a tensor value
    return last_term.item()  

################################################################################################

# def compute_grad_variance(grads):  #list of lists consisting of gradient values....
#     # print (grads)
#     """E[|| g ||^2 ]"""
#     # total_ = 0
#     # for i in range(len(grads)):
#     #     # grads[i] = torch.tensor(grads[i])
#     #     val = torch.norm(grads[i])
#     #     total_ += val
#     grads = 
#     ft = total_ / len(grads)
    
#     """|| E[g] ||^2"""
#     # grads = torch.tensor(grads)
#     # all_grads = torch.zeros(len(grads[0]))
#     # for i in grads:
#       # all_grads += i 
    
#     # avg_grads = all_grads / len(grads)
#     # print (grads.size())
#     avg_grads = torch.mean(grads, 0)
#     st = torch.norm(avg_grads)
    
#     variance_score = ft - st
#     return variance_score
