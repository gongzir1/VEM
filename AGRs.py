from args import args
from eval import *
from misc import *
import torch
import pickle
import torch.nn as nn
import argparse, os, sys, csv, shutil, time, random, operator, pickle, ast, math, copy
import numpy as np
from utils import *

def cosine_distance_PICK_OUT(all_updates,FLmodel,n_attackers):
    cs_selected_all={}
    cs_selected,cs_not_selected=compare_user_updates_cluster(FLmodel, all_updates,n_attackers)

    return cs_selected,cs_not_selected

def cosine_distance(all_updates,FLmodel,n_attackers):
    cs_selected_all={}
    cs_selected=compare_user_updates_layer_wise_new(FLmodel, all_updates)

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            cs_selected_all[str(n)]=all_updates[str(n)][cs_selected, :]

    return cs_selected_all
            
def my_cosine_distance(all_updates,FLmodel,n_attackers):
    cs_selected_all={}
    cs_selected=my_compare_user_updates_cluster(FLmodel, all_updates)

    # for n, m in FLmodel.named_modules():
    #     if hasattr(m, "scores"): 
    #         cs_selected_all[str(n)]=all_updates[str(n)][cs_selected, :]
    

    # return cs_selected_all
    
def tr_mean(all_updates, n_attackers):
    sorted_updates = torch.sort(all_updates, 0)[0]
    out = torch.mean(sorted_updates[n_attackers:-n_attackers], 0) if n_attackers else torch.mean(sorted_updates,0)
    return out

def multi_krum(all_updates, n_attackers, multi_k=False):

    candidates = []
    candidate_indices = []
    remaining_updates = all_updates
    all_indices = np.arange(len(all_updates))

    while len(remaining_updates) > 2 * n_attackers + 2:
        torch.cuda.empty_cache()
        distances = []
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break
    # print(len(remaining_updates))

    aggregate = torch.mean(candidates, dim=0)

    return aggregate, np.array(candidate_indices)