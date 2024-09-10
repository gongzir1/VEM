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

def plot(FLmodel, user_updates,i):
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            if str(n)=='linear.4':
                args_sorts=torch.sort(user_updates[str(n)])[1]
                ####plot########
                args_sorts_cpu = args_sorts.cpu()
                
                # first_row = args_sorts_cpu[0]

                # Plot the first row values against the index
                plt.plot(args_sorts_cpu[0].numpy(),marker='o', markersize=2,linestyle='None',)
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.title('Plot of the first row of the tensor')
                # plt.show()
                # plt.savefig('plot.pdf')
                filename = f'plot_iteration_{i}.pdf'  # Change filename for each iteration
                plt.savefig(filename)
                plt.close()
                
                ####################
                del args_sorts,args_sorts_cpu
                break

# def FRL_Vote(FLmodel, user_updates, initial_scores,mylist,agr_matrix):
#     for n, m in FLmodel.named_modules():
#         if hasattr(m, "scores"):
#             if str(n) in mylist:  
#                 # #################og######################
#                 # args_sorts=torch.sort(user_updates[str(n)])[1]
#                 # sum_args_sorts=torch.sum(args_sorts, 0)         
#                 # idxx=torch.sort(sum_args_sorts)[1]    
#                 # # print("og ranks is")
#                 # # print(idxx)
#                 # ############## select max ##############################        
#                 id_len = torch.zeros(agr_matrix[str(n)].size(0), dtype=torch.int32)

#                 # Iterate through the rows in reverse order
#                 # for i in range(agr_matrix[str(n)].size(0) - 1, -1, -1):
#                 #     # Find the index of the largest value in the current row
#                 #     idx_largest = torch.argmax(agr_matrix[str(n)][i])
                    
#                 #     # Set the values of the column corresponding to the largest value to 0
#                 #     agr_matrix[str(n)][:, idx_largest] = 0
                    
#                 #     # Store the index of the largest value
#                 #     idxx[i] = idx_largest
#                 ##############################################################
#                 ##################### same as og###############################
#                 A=torch.arange(0,len(id_len),dtype=torch.int32)
#                 w=torch.matmul(A,agr_matrix[str(n)])
#                 idxx=torch.sort(w)[1]  
#                 # print('new is:')
#                 # print(idxx)

#                 temp1=m.scores.detach().clone()
#                 temp1.flatten()[idxx]=initial_scores[str(n)] # assign the score based on ranking
#                 m.scores=torch.nn.Parameter(temp1)                       
#                 del idxx, temp1
#             else:
#                 args_sorts=torch.sort(user_updates[str(n)])[1]
#                 sum_args_sorts=torch.sum(args_sorts, 0)         
#                 idxx=torch.sort(sum_args_sorts)[1]          # get the rank again
#                 temp1=m.scores.detach().clone()
#                 temp1.flatten()[idxx]=initial_scores[str(n)] # assign the score based on ranking
#                 m.scores=torch.nn.Parameter(temp1)                       
#                 del idxx, temp1

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

def sep(FLmodel, user_updates):
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            if str(n)=='convs.0':                                                           # only record the first layer
                args_sorts=torch.sort(user_updates[str(n)])[1]
                sum_args_sorts=torch.sum(args_sorts, 0)      
                idxx=torch.sort(sum_args_sorts)[1]          # get the rank again
                # record the top k% edge id
                id_selected=idxx[len(idxx)//2:]
                id_not_selected=idxx[:len(idxx)//2]
    
    return id_selected, id_not_selected

def update_and_record_crossings(FLmodel, user_updates,selected_old, not_selected_old, crossings):
    # Convert current states to sets for efficient membership checking
    selected_old_set, not_selected_old_set = set(selected_old.cpu().numpy()), set(not_selected_old.cpu().numpy())
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            if str(n)=='convs.0':                                                           # only record the first layer
                args_sorts=torch.sort(user_updates[str(n)])[1]
                sum_args_sorts=torch.sum(args_sorts, 0)      
                idxx=torch.sort(sum_args_sorts)[1]          # get the rank again
                # record the top k% edge id
                id_selected=set((idxx[len(idxx)//2:]).cpu().numpy())
                id_not_selected=set((idxx[:len(idxx)//2]).cpu().numpy())

    total = set(range(len(idxx)))

    # Record crossings for each item
    # for item in total:
    #     if (item in not_selected_old_set and item in id_selected) or (item in selected_old_set and item in id_not_selected):
    #         crossings[item] += 1
    for item in total:
        if item in selected_old_set and item in id_not_selected:
            crossings[item] += 1
    return crossings
               

def FRL_Vote_adaptive(FLmodel, user_updates, initial_scores,k_a):
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            args_sorts=torch.sort(user_updates[str(n)])[1]
            sum_args_sorts=torch.sum(k_a*args_sorts, 0)         
            idxx=torch.sort(sum_args_sorts)[1]          # get the rank again
            temp1=m.scores.detach().clone()
            temp1.flatten()[idxx]=initial_scores[str(n)] # assign the score based on ranking
            m.scores=torch.nn.Parameter(temp1)                       
            del idxx, temp1


def FRL_Vote_both(FLmodel, user_updates, initial_scores):
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            args_sorts=torch.sort(user_updates[str(n)])[1]
            sum_args_sorts=torch.sum(args_sorts, 0)         
            idxx=torch.sort(sum_args_sorts)[1]          # get the rank again
            # calculate intersection
            # if str(n)=='convs.0':
            intersection=0

            ############plot #############
            top_k_mal=torch.topk(torch.sort(idxx)[1],len(idxx)//10)[1]
            ############################
            ##########benign ##########
            user_updates_b=user_updates[str(n)][:20]
            args_sorts_b=torch.sort(user_updates_b)[1]
            sum_args_sorts_b=torch.sum(args_sorts_b, 0)         
            idxx_b=torch.sort(sum_args_sorts_b)[1]          # get the rank again
            ############plot #############
            top_k_b=torch.topk(torch.sort(idxx_b)[1],len(idxx)//10)[1]
            ################intersection#############
            # Convert tensors to sets
            set1 = set(top_k_b.tolist())
            set2 = set(top_k_mal.tolist())

            # Find intersection
            intersection = set1.intersection(set2)

            # Get the length of the intersection
            intersection_length = len(intersection)
            # succ=len(idxx)//10-intersection_length
            not_change=intersection_length
            succ_rate=1-not_change/(len(idxx)//10)

            # with (args.run_base_dir / "inter.txt").open("a") as f:
            #     f.write("\n" + str(intersection_length) + ", " + str(succ) + ", " + str(succ_rate))
            with (args.run_base_dir / "inter.txt").open("a") as f:
                f.write("\n"+str(succ_rate))
            # wandb.log(
            # {
            #     "inter": intersection_length,
            #     "succ":succ,
            #     "succ_rate":succ_rate

            # })
            
            temp1=m.scores.detach().clone()
            temp1.flatten()[idxx]=initial_scores[str(n)] # assign the score based on ranking
            m.scores=torch.nn.Parameter(temp1)                       
            del idxx, temp1
    return succ_rate
            
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

def train_backdoor(trainloader, model, criterion, optimizer, device):
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


def compare_user_updates(FLmodel, user_updates):
    similarities = []
    concatenated_updates_tensor = []

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            if len(concatenated_updates_tensor) == 0: # Check if concatenated_updates_tensor is empty
                concatenated_updates_tensor = user_updates[str(n)].float()
            else:
                concatenated_updates_tensor = torch.stack([torch.cat((row_a, row_b)) for row_a, row_b in zip(concatenated_updates_tensor, user_updates[str(n)])])

    num_rows = concatenated_updates_tensor.shape[0]
    for i in range(num_rows):
        for j in range(i+1, num_rows):
            row_i = concatenated_updates_tensor[i]
            row_j = concatenated_updates_tensor[j]
            similarity = F.cosine_similarity(row_i, row_j, dim=0)
            similarities.append(similarity.item())

    # similarities = np.array(similarities)
    # similarities = (similarities - np.min(similarities)) / (np.max(similarities) - np.min(similarities))

    # Plot similarities
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(similarities)), similarities, marker='.')
    plt.title('Similarities')
    plt.xlabel('Pair Index')
    plt.ylabel('Similarity')
    plt.grid(True)
    plt.show()

    return similarities

def compare_user_updates_cluster(FLmodel, user_updates,n_attackers):
    similarities = []
    concatenated_updates_tensor = []

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            if len(concatenated_updates_tensor) == 0: # Check if concatenated_updates_tensor is empty
                concatenated_updates_tensor = user_updates[str(n)].float()
            else:
                concatenated_updates_tensor = torch.stack([torch.cat((row_a, row_b)) for row_a, row_b in zip(concatenated_updates_tensor, user_updates[str(n)])])

    x1 = concatenated_updates_tensor
    x2 =concatenated_updates_tensor
    eps=1e-8
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    similarities = torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
    k=len(concatenated_updates_tensor)-n_attackers


    k_nearest = torch.topk(similarities, k, dim=1)      # return the largest for each client 
    neighbour_dist = torch.zeros(concatenated_updates_tensor.size(0))
    for i in range(concatenated_updates_tensor.size(0)):
        idx = k_nearest.indices[i]
        neighbour = similarities[idx][:,idx]
        neighbour_dist[i] = neighbour.sum()

    all_indices = torch.arange(neighbour_dist.size(0)) 
    cos_selected = torch.topk(neighbour_dist, k).indices
    # print(cos_selected)
    return cos_selected

def compare_user_updates_ranked_cluster(FLmodel, user_updates,n_attackers):
    similarities = []
    concatenated_updates_tensor = []

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            args_sorts=torch.sort(user_updates[str(n)])[1]
            # sum_args_sorts=torch.sum(args_sorts, 0)   
            if len(concatenated_updates_tensor) == 0: # Check if concatenated_updates_tensor is empty
                concatenated_updates_tensor = args_sorts.float()
            else:
                concatenated_updates_tensor = torch.stack([torch.cat((row_a, row_b)) for row_a, row_b in zip(concatenated_updates_tensor, args_sorts)])

    x1 = concatenated_updates_tensor
    x2 =concatenated_updates_tensor
    eps=1e-8
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    similarities = torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
    k=len(concatenated_updates_tensor)-n_attackers

    k_nearest = torch.topk(similarities, k, dim=1)      # return the largest for each client 
    neighbour_dist = torch.zeros(concatenated_updates_tensor.size(0))
    for i in range(concatenated_updates_tensor.size(0)):
        idx = k_nearest.indices[i]
        neighbour = similarities[idx][:,idx]
        neighbour_dist[i] = neighbour.sum()
    cos_selected = torch.topk(neighbour_dist, k).indices
    # print(cos_selected)
    return cos_selected

def compare_user_updates_layer_wise(FLmodel, user_updates):
    similarities = []
    concatenated_updates_tensor = []


    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            # if len(concatenated_updates_tensor) == 0: # Check if concatenated_updates_tensor is empty
            #     concatenated_updates_tensor = user_updates[str(n)].float()
            # else:
            #     concatenated_updates_tensor = torch.stack([torch.cat((row_a, row_b)) for row_a, row_b in zip(concatenated_updates_tensor, user_updates[str(n)])])

            num_rows = user_updates[str(n)].shape[0]
            for key in user_updates.keys():
                value = user_updates[key]
                # half
                # Calculate the number of elements to select
                num_elements = value.size(1)
                first_half = value[:, :num_elements]


                for i in range(num_rows):
                    for j in range(i+1, num_rows):
                        row_i = first_half[i].float()
                        row_j = first_half[j].float()
                        similarity = F.cosine_similarity(row_i, row_j, dim=0)
                        similarities.append(similarity.item())

            # similarities = np.array(similarities)
            # similarities = (similarities - np.min(similarities)) / (np.max(similarities) - np.min(similarities))

                # Plot similarities
            plt.figure(figsize=(8, 6))
            plt.scatter(range(len(similarities)), similarities, marker='.')
            plt.title('Similarities')
            plt.xlabel('Pair Index')
            plt.ylabel('Similarity')
            plt.grid(True)
            # Add vertical lines at every 10th index
            for i in range(10, len(similarities), 10):
                plt.axvline(x=i, color='black', linestyle='--')
            plt.show()


    return similarities

def compare_user_updates_distance(FLmodel, user_updates):
    distances = []

    concatenated_updates_tensor = []

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            if len(concatenated_updates_tensor) == 0: # Check if concatenated_updates_tensor is empty
                concatenated_updates_tensor = user_updates[str(n)].float()
            else:
                concatenated_updates_tensor = torch.stack([torch.cat((row_a, row_b)) for row_a, row_b in zip(concatenated_updates_tensor, user_updates[str(n)])])

    new_tensor = torch.unsqueeze(concatenated_updates_tensor, dim=0).cpu()
    new_tensor_2 = torch.unsqueeze(concatenated_updates_tensor, dim=1).cpu()

    distance = torch.abs(new_tensor - new_tensor_2).float()

    mean_along_third_dim = torch.mean(distance, dim=2)
    non_zero_tensor = mean_along_third_dim[mean_along_third_dim != 0]
    mean_along_third_dim_normalized = (non_zero_tensor - non_zero_tensor.mean()) / non_zero_tensor.std()
    

    distances=mean_along_third_dim_normalized.flatten()
    distances_numpy = distances.detach().numpy()


    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(distances_numpy)), distances_numpy, marker='o', color='b')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Normalized Tensor')
    plt.grid(True)
    plt.show()

    return distances

def my_compare_user_updates_cluster(FLmodel, user_updates):
    similarities = []
    concatenated_updates_tensor = []

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            if len(concatenated_updates_tensor) == 0: # Check if concatenated_updates_tensor is empty
                concatenated_updates_tensor = user_updates[str(n)].float()
            else:
                concatenated_updates_tensor = torch.stack([torch.cat((row_a, row_b)) for row_a, row_b in zip(concatenated_updates_tensor, user_updates[str(n)])])

    x1 = concatenated_updates_tensor
    x2 =concatenated_updates_tensor
    eps=1e-8
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    similarities = torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
    # k=len(concatenated_updates_tensor)-n_attackers


    # k_nearest = torch.topk(similarities, k, dim=1)      # return the largest for each client 
    # neighbour_dist = torch.zeros(concatenated_updates_tensor.size(0))
    # for i in range(concatenated_updates_tensor.size(0)):
    #     idx = k_nearest.indices[i]
    #     neighbour = similarities[idx][:,idx]
    #     neighbour_dist[i] = neighbour.sum()
    # cos_selected = torch.topk(neighbour_dist, k).indices

    max_clusters_to_try=5
    # Compute silhouette scores for different numbers of clusters
    silhouette_scores = []
    similarities_cpu = similarities.cpu().detach().numpy()

    for k in range(2, max_clusters_to_try):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(similarities_cpu)
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(similarities_cpu, labels))

    # Plot silhouette scores
    # plt.plot(range(2, max_clusters_to_try), silhouette_scores, marker='o')
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Silhouette Score')
    # plt.title('Silhouette Score vs Number of Clusters')
    # plt.show()

    optimal_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # Add 2 to account for the range starting from 2

    # Perform clustering
    kmeans = KMeans(n_clusters=optimal_n_clusters)  # Set the number of clusters
    kmeans.fit(similarities_cpu)
    clusters = kmeans.predict(similarities_cpu)

    # Plot the clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(similarities_cpu[:, 0], similarities_cpu[:, 1], c=clusters, cmap='viridis')
    plt.title('Clusters of Similarities')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.show()
    


    # return cos_selected
def compare_user_updates_layer_wise_new(user_updates):
    similarities = []

    # num_rows = user_updates[1].shape[0]
    num_rows = next(iter(user_updates.values())).shape[0]

    for key in user_updates.keys():
        value = user_updates[key]
            
        num_elements = value.size(1)
        p=0.0625
        num_indices_to_select = int(value.size(1) * p)
        # Calculate the start and end indices for selecting the middle p% indices
        start_index = (value.size(1) - num_indices_to_select) // 2
        end_index = start_index + num_indices_to_select
        # first_half = value[:, :num_elements]
        selected_indices = value[:,start_index:end_index]

       

        for i in range(num_rows):
            for j in range(i+1, num_rows):
                row_i = selected_indices[i].float()
                row_j = selected_indices[j].float()
                similarity = F.cosine_similarity(row_i, row_j, dim=0)
                similarities.append(similarity.item())

            
                # Plot similarities
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(similarities)), similarities, marker='.')
    plt.title('Similarities')
    plt.xlabel('Pair Index')
    plt.ylabel('Similarity')
    plt.grid(True)
    # Add vertical lines at every 10th index
    for i in range(10, len(similarities), 10):
        plt.axvline(x=i, color='black', linestyle='--')
    plt.show()

    del similarities

    # return similarities

def compare_user_updates_ranked_cluster_layer_wise(FLmodel, user_updates):
    similarities = []
    concatenated_updates_tensor = []

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            args_sorts=torch.sort(user_updates[str(n)])[1]

            num_rows = args_sorts.shape[0]

            num_elements = args_sorts.size(1)
            p=0.5
            num_indices_to_select = int(args_sorts.size(1) * p)
            # Calculate the start and end indices for selecting the middle p% indices
            start_index = (args_sorts.size(1) - num_indices_to_select) // 2
            end_index = start_index + num_indices_to_select
            # first_half = value[:, :num_elements]
            selected_indices = args_sorts[:,start_index:end_index]


            for i in range(num_rows):
                for j in range(i+1, num_rows):
                    row_i = args_sorts[i].float()
                    row_j = args_sorts[j].float()
                    similarity = F.cosine_similarity(row_i, row_j, dim=0)
                    similarities.append(similarity.item())

            # similarities = np.array(similarities)
            # similarities = (similarities - np.min(similarities)) / (np.max(similarities) - np.min(similarities))

                # Plot similarities
            plt.figure(figsize=(8, 6))
            plt.scatter(range(len(similarities)), similarities, marker='.')
            plt.title('Similarities')
            plt.xlabel('Pair Index')
            plt.ylabel('Similarity')
            plt.grid(True)
            # Add vertical lines at every 10th index
            for i in range(10, len(similarities), 10):
                plt.axvline(x=i, color='black', linestyle='--')
            plt.show()
        # print('finish')


