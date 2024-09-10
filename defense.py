from args import args
from eval import *
from misc import *
import torch
from AGRs import *
import torch.nn as nn
import argparse, os, sys, csv, shutil, time, random, operator, pickle, ast, math, copy
import numpy as np
import utils
from collections import defaultdict
import wandb
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import defaultdict
import collections



def Euclidean(FLmodel, user_updates,n_attackers):
    distance = []
    concatenated_updates_tensor = []

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            if len(concatenated_updates_tensor) == 0: # Check if concatenated_updates_tensor is empty
                concatenated_updates_tensor = user_updates[str(n)].float()
            else:
                concatenated_updates_tensor = torch.stack([torch.cat((row_a, row_b)) for row_a, row_b in zip(concatenated_updates_tensor, user_updates[str(n)])])

    dist_matrix = torch.cdist(concatenated_updates_tensor, concatenated_updates_tensor)
    
    k=len(concatenated_updates_tensor)-n_attackers

    k_nearest = torch.topk(dist_matrix, k, largest=False,dim=1)      # return the largest for each client 
    neighbour_dist = torch.zeros(concatenated_updates_tensor.size(0))
    for i in range(concatenated_updates_tensor.size(0)):
        idx = k_nearest.indices[i]
        neighbour = dist_matrix[idx][:,idx]
        neighbour_dist[i] = neighbour.sum()

    all_indices = torch.arange(neighbour_dist.size(0)) 
    eud_selected = torch.topk(neighbour_dist, k,largest=False).indices

    selected_dict = defaultdict(lambda: torch.empty(len(eud_selected), user_updates[next(iter(user_updates))].size(1)))

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            selected_dict[n]=user_updates[str(n)][eud_selected]
    return selected_dict

def Krum(FLmodel, user_updates, n_attackers):
    distance = []
    concatenated_updates_tensor = []

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            if len(concatenated_updates_tensor) == 0:  # Check if concatenated_updates_tensor is empty
                concatenated_updates_tensor = user_updates[str(n)].float()
            else:
                concatenated_updates_tensor = torch.stack([torch.cat((row_a, row_b)) for row_a, row_b in zip(concatenated_updates_tensor, user_updates[str(n)])])

    dist_matrix = torch.cdist(concatenated_updates_tensor, concatenated_updates_tensor)
    
    k = len(concatenated_updates_tensor) - n_attackers - 2  # N - m - 2

    k_nearest = torch.topk(dist_matrix, k + 1, largest=False, dim=1)  # Get k+1 nearest because the closest will include itself
    neighbour_dist = torch.zeros(concatenated_updates_tensor.size(0))
    
    for i in range(concatenated_updates_tensor.size(0)):
        idx = k_nearest.indices[i][1:]  # Exclude the closest one (itself)
        neighbour = dist_matrix[i][idx]
        neighbour_dist[i] = neighbour.sum()

    all_indices = torch.arange(neighbour_dist.size(0)) 
    selected_index = torch.argmin(neighbour_dist)  # Index of the client with the smallest neighbor distance

    selected_dict = defaultdict(lambda: torch.empty(1, user_updates[next(iter(user_updates))].size(1)))

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            selected_dict[n] = user_updates[str(n)][selected_index:selected_index+1]  # Select only the closest one
    
    return selected_dict

def FRL_fang(FLmodel, user_updates,round_users,n_attackers,val_loader,criterion,device,initial_scores,mode):
    losses_list = []
    acc_list = []
    k=round_users-n_attackers
    for i in range(round_users):
        local_rank=user_updates[i]

    with torch.no_grad():
        for batch_ind, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device, torch.float)
            targets = targets.to(device, torch.long)
            #   get local model
            local_models=utils.Get_local_models(FLmodel, user_updates, initial_scores,round_users)
            for i in range(round_users):
                outputs = local_models[i](inputs)
                if len(outputs.shape) == 1:
                    outputs = outputs.unsqueeze(0)

                loss = criterion(outputs, targets)
                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

                # Append individual losses and top-1 accuracies to lists
                losses_list.append(loss.data)
                acc_list.append(prec1 / 100.0)
        # Convert lists to tensors
        losses = torch.tensor(losses_list)
        acc = torch.tensor(acc_list)

    selected_dict = defaultdict(lambda: torch.empty(k, user_updates[next(iter(user_updates))].size(1)))
    ERR = torch.topk(acc, k).indices
    LFR = torch.topk(losses, k, largest=False).indices
    
    if mode == "ERR":  # acc
        fang_selected=ERR
    elif mode =='LFR':  # loss
        fang_selected = LFR
    elif mode =='combined':
        union = torch.cat([ERR, LFR])
        uniques, counts = union.unique(return_counts=True)
        fang_selected = uniques[counts > 1]

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            selected_dict[n]=user_updates[str(n)][fang_selected]
    return selected_dict


def fl_trust(FLmodel, user_updates,round_users,n_attackers,val_loader,criterion,device,initial_scores,e):
    k=round_users-n_attackers
    b_update=collections.defaultdict(list)
    b_rank_cat=[]
    # get beign reference ranking
    mp = copy.deepcopy(FLmodel)
    optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
    for epoch in range(args.local_epochs):
        train_loss, train_acc = train(val_loader, mp, criterion, optimizer, args.device)
        scheduler.step()
    
    for n, m in mp.named_modules():
        if hasattr(m, "scores"):
            b_rank=Find_rank(m.scores.detach().clone())
            b_update[str(n)]=b_rank[None,:] if len(b_update[str(n)]) == 0 else torch.cat((b_update[str(n)], b_rank[None,:]), 0)
            del b_rank

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            if len(b_rank_cat) == 0: # Check if concatenated_updates_tensor is empty
                b_rank_cat = b_update[str(n)].float()
            else:
                b_rank_cat = torch.stack([torch.cat((row_a, row_b)) for row_a, row_b in zip(b_rank_cat, b_update[str(n)])])
    # concat user updates

    similarities = []
    concatenated_updates_tensor = []

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            if len(concatenated_updates_tensor) == 0: # Check if concatenated_updates_tensor is empty
                concatenated_updates_tensor = user_updates[str(n)].float()
            else:
                concatenated_updates_tensor = torch.stack([torch.cat((row_a, row_b)) for row_a, row_b in zip(concatenated_updates_tensor, user_updates[str(n)])])

    x1 = b_rank_cat
    x2 =concatenated_updates_tensor
    eps=1e-8
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = x2.norm(p=2, dim=1, keepdim=True)
    similarities = torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
    relu = torch.nn.ReLU()
    norm = b_rank_cat.norm()
    scores=relu(similarities)
    
    #######majority voting###########
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            args_sorts=torch.sort(scores.transpose(0,1)*user_updates[str(n)])[1]
            sum_args_sorts=torch.sum(args_sorts, 0)         
            idxx=torch.sort(sum_args_sorts)[1]          # get the rank again
            temp1=m.scores.detach().clone()
            temp1.flatten()[idxx]=initial_scores[str(n)] # assign the score based on ranking
            m.scores=torch.nn.Parameter(temp1)                       
            del idxx, temp1


def FABA(FLmodel, user_updates, n_attackers):
    concatenated_updates_tensor = []
    all_updates = []

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            update_tensor = user_updates[str(n)].float()
           
            if len(concatenated_updates_tensor) == 0:
                concatenated_updates_tensor = update_tensor
            else:
                concatenated_updates_tensor = torch.stack([torch.cat((row_a, row_b)) for row_a, row_b in zip(concatenated_updates_tensor, update_tensor)])

    # Calculate the mean of all user updates
    mean_update = torch.mean(concatenated_updates_tensor, dim=0)

    # Calculate the distance of each update from the mean
    distances = torch.cdist(concatenated_updates_tensor.unsqueeze(0), mean_update.unsqueeze(0)).squeeze(0).squeeze()

    # Number of updates to select (excluding the attackers)
    k = len(concatenated_updates_tensor) - n_attackers

    # Select the k smallest distances
    selected_indices = torch.topk(distances, k, largest=False).indices

    # Create a dictionary to store the selected updates
    selected_dict = defaultdict(lambda: torch.empty(len(selected_indices), user_updates[next(iter(user_updates))].size(1)))

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            selected_dict[n] = user_updates[str(n)][selected_indices]

    return selected_dict


def DnC(FLmodel, user_updates, num_byzantine: int, sub_dim, num_iters, filter_frac) -> torch.Tensor:
    """
    A robust aggregator from the paper `Manipulating the Byzantine: Optimizing
    Model Poisoning Attacks and Defenses for Federated Learning.
    
    <https://par.nsf.gov/servlets/purl/10286354>.

    Args:
        inputs (List[torch.Tensor]): List of update tensors from different clients.
        num_byzantine (int): Number of Byzantine (malicious) clients.
        sub_dim (int): Dimensionality of the subspace used in each iteration. Default is 10000.
        num_iters (int): Number of iterations to perform the aggregation. Default is 5.
        filter_frac (float): Fraction of updates to filter out as potential Byzantine. Default is 1.0.

    Returns:
        torch.Tensor: The robustly aggregated update.
    """
    updates = []
    # all_updates = []
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            update_tensor = user_updates[str(n)].float()
           
            if len(updates) == 0:
                updates = update_tensor
            else:
                updates = torch.stack([torch.cat((row_a, row_b)) for row_a, row_b in zip(updates, update_tensor)])


    d = len(updates[0])

    benign_ids = []
    for i in range(num_iters):
        indices = torch.randperm(d)[:sub_dim]
        sub_updates = updates[:, indices]
        mu = sub_updates.mean(dim=0)
        centered_update = sub_updates - mu
        v = torch.linalg.svd(centered_update, full_matrices=False)[2][0, :]
        s = np.array(
            [(torch.dot(update - mu, v) ** 2).item() for update in sub_updates]
        )

        good = s.argsort()[:len(updates) - int(filter_frac * num_byzantine)]
        benign_ids.append(good)

    # Convert the first list to a set to start the intersection
    intersection_set = set(benign_ids[0])

    # Iterate over the rest of the lists and get the intersection
    for lst in benign_ids[1:]:
        intersection_set.intersection_update(lst)

    # Convert the set back to a list
    benign_ids = list(intersection_set)
    selected_dict = defaultdict(lambda: torch.empty(len(benign_ids), user_updates[next(iter(user_updates))].size(1)))

    # benign_updates = updates[benign_ids, :]
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            selected_dict[n]=user_updates[str(n)][benign_ids]
    return selected_dict

def foolsgold(FLmodel, user_updates,device,initial_scores):
    similarities = []
    concatenated_updates_tensor = []
    user_updates_new=collections.defaultdict(list)

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            if len(concatenated_updates_tensor) == 0: # Check if concatenated_updates_tensor is empty
                concatenated_updates_tensor = user_updates[str(n)].float()
            else:
                concatenated_updates_tensor = torch.stack([torch.cat((row_a, row_b)) for row_a, row_b in zip(concatenated_updates_tensor, user_updates[str(n)])])

    x1 = concatenated_updates_tensor
    x2 =concatenated_updates_tensor
    eps=1e-4
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    n_clients = concatenated_updates_tensor.shape[0]

    cs = torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
    cs = cs - torch.eye(n_clients, device=device)

    # cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    maxcs = torch.max(cs, dim=1).values

    # Pardoning step
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i, j] = cs[i, j] * maxcs[i] / maxcs[j]

    wv = 1 - torch.max(cs, dim=1).values
    wv = torch.clamp(wv, 0, 1)

    # Rescale so that max value is 0.99
    wv = wv / torch.max(wv)
    wv[wv == 1] = 0.99
    wv[wv==0]=0.1

    # Logit function
    # wv = torch.log(wv / (1 - wv)+eps) + 0.5  # Added eps to avoid log(0)
    # wv = torch.clamp(wv, 0, 1)
    # wv=wv/torch.sum(wv).item()
    wv=wv.view(25,1)
    #######majority voting###########
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            args_sorts=torch.sort(wv*user_updates[str(n)])[1]
            sum_args_sorts=torch.sum(args_sorts, 0)         
            idxx=torch.sort(sum_args_sorts)[1]          # get the rank again
            temp1=m.scores.detach().clone()
            temp1.flatten()[idxx]=initial_scores[str(n)] # assign the score based on ranking
            m.scores=torch.nn.Parameter(temp1)                       
            del idxx, temp1
