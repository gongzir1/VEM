from args import args
from eval import *
from misc import *
import torch
import pickle
import torch.nn as nn
import argparse, os, sys, csv, shutil, time, random, operator, pickle, ast, math, copy
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns


def min_max(scores,device):
    deviation = torch.std(scores, 0).to(device)       # deviation of each dimension
    model_re=scores[0].to(device)         # set the first beign score as the base
    lamda = torch.Tensor([10.0]).float().to(device) 
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0
    distance = torch.cdist(scores, scores)
    max_distance = torch.max(distance)
    del distance
    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        distance = torch.norm((scores - mal_update), dim=1) ** 2
        max_d = torch.max(distance)
        if max_d <= max_distance:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2
        lamda_fail = lamda_fail / 2
    mal_update = (model_re - lamda_succ * deviation)
    return mal_update


def min_sum(scores,device):
    deviation = torch.std(scores, 0).to(device)       # deviation of each dimension
    model_re=scores[0].to(device)         # set the first beign score as the base
    lamda = torch.Tensor([10.0]).float().to(device) 
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0
    distance = torch.cdist(scores, scores)
    sum_dis=torch.sum(distance,0)
    min_distance=torch.min(sum_dis)
    del distance
    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        distance = torch.norm((scores - mal_update), dim=1) ** 2
        # max_d = torch.max(distance)
        score_sum = torch.sum(distance)
        if score_sum <= min_distance:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2
        lamda_fail = lamda_fail / 2
    mal_update = (model_re - lamda_succ * deviation)
    return mal_update

def fang_attack(scores,device):
    maximum_dim, _ = torch.max(scores, dim=0) # get the max value for each dimension
    minimum_dim, _ = torch.min(scores, dim=0)

    direction = torch.sign(torch.sum(scores, dim=0))

    directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim 

    
    random_12 = (1. + torch.rand(scores.size(1))).to(device)
    mal_score = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
    return mal_score



