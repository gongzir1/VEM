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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import wandb


def Find_rank(scores):
    _, idx = scores.detach().flatten().sort()
    return idx.detach()
def Find_rank_attack(scores):
    _, idx = scores.detach().sort()
    return idx.detach()

def FRL_Vote(FLmodel, user_updates, initial_scores):
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            args_sorts=torch.sort(user_updates[str(n)])[1]
            sum_args_sorts=torch.sum(args_sorts, 0)         
            idxx=torch.sort(sum_args_sorts)[1]          # get the rank again
            temp1=m.scores.detach().clone()
            temp1.flatten()[idxx]=initial_scores[str(n)] # assign the score based on ranking
            m.scores=torch.nn.Parameter(temp1)                       
            del idxx, temp1
             
        
def Get_local_models(FLmodel, user_updates, initial_scores,users):
    local_models = []
    users=25
        # Iterate over each row in idxx_new
    for row_idx in range(users):
        # Create a copy of the FLmodel
        local_model = copy.deepcopy(FLmodel)

        for n, m in local_model.named_modules():
            if hasattr(m, "scores"):
                # idxx = torch.tensor(user_updates[str(n)][row_idx], dtype=torch.long) 
                idxx = user_updates[str(n)][row_idx].clone().detach().to(torch.long)     # get the rank again
                temp1=m.scores.detach().clone()
                temp1.flatten()[idxx]=initial_scores[str(n)] # assign the score based on ranking
                m.scores=torch.nn.Parameter(temp1)                       
                del idxx, temp1
            
        # Add the local model to the list
        local_models.append(local_model)
    
    return local_models
def Get_group_models(FLmodel, user_updates, initial_scores):
    group_model = copy.deepcopy(FLmodel)
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            args_sorts=torch.sort(user_updates[str(n)])[1]
            sum_args_sorts=torch.sum(args_sorts, 0)         
            idxx=torch.sort(sum_args_sorts)[1]          # get the rank again

            temp1=m.scores.detach().clone()
            temp1.flatten()[idxx]=initial_scores[str(n)] # assign the score based on ranking
            m.scores=torch.nn.Parameter(temp1)                       
            del idxx, temp1                      
    return group_model
            
def train_label_flip(trainloader, model, criterion, optimizer, device):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_ind, (inputs, targets) in enumerate(trainloader):

        inputs = inputs.to(device, torch.float)
        targets = targets.to(device, torch.long)

         # Change the labels from l to L - l - 1 label flip attack
        targets_flipped = 10 - targets - 1  # untarget label flip
    
        # Change the labels to a specific value (e.g., 1)
        # target label flip
        # targets_flipped = torch.full_like(targets, 1, device=device, dtype=torch.long)


        # targets_flipped = targets_flipped.to(device, torch.long)

        # Print original and flipped labels
        # print('Original Labels:', targets)
        # print('Flipped Labels:', targets_flipped)

        outputs = model(inputs)
        if len(outputs.shape) == 1:
            outputs = outputs.unsqueeze(0)
        loss = criterion(outputs, targets_flipped)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets_flipped.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size()[0])
        top1.update(prec1.item()/100.0, inputs.size()[0])
        top5.update(prec5.item()/100.0, inputs.size()[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return (losses.avg, top1.avg)
        
def train(trainloader, model, criterion, optimizer, device):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_ind, (inputs, targets) in enumerate(trainloader):

        inputs = inputs.to(device, torch.float)
        targets = targets.to(device, torch.long)

        outputs = model(inputs)
        if len(outputs.shape) == 1:
            outputs = outputs.unsqueeze(0)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size()[0])
        top1.update(prec1.item()/100.0, inputs.size()[0])
        top5.update(prec5.item()/100.0, inputs.size()[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        for batch_ind, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device, torch.float)
            targets = targets.to(device, torch.long)
            outputs = model(inputs)
            if len(outputs.shape) == 1:
                outputs = outputs.unsqueeze(0)
            
            loss = criterion(outputs, targets)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data, inputs.size()[0])
            top1.update(prec1/100.0, inputs.size()[0])
            top5.update(prec5/100.0, inputs.size()[0])
    return (losses.avg, top1.avg)


