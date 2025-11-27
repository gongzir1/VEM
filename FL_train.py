from args import args
import torch
import torch.nn as nn
import models
from utils import *
from AGRs import *
import copy
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import defaultdict
import collections
import wandb
import VEM
import matplotlib.pyplot as plt
import defense
import other_attacks


def FRL_VEM(tr_loaders, te_loader):
    print ("#########FRL under VEM attack############")


    args.conv_type = 'MaskConv'
    args.conv_init = 'signed_constant'
    args.bn_type="NonAffineNoStatsBN"    
    estimation='his'

    
    n_attackers = int(args.nClients * args.at_fractions)
    sss = "fraction of maliciou clients: %.2f | total number of malicious clients: %d"%(args.at_fractions,
                                                                                        n_attackers)
    print (sss)
    with (args.run_base_dir / "output.txt").open("a") as f:
        f.write("\n"+str(sss))
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    FLmodel = getattr(models, args.model)().to(args.device)
    
    initial_scores={}
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            initial_scores[str(n)]=m.scores.detach().clone().flatten().sort()[0]
    
    e=0
    t_best_acc=0
    Rg_hist =collections.defaultdict(list)
    Rm_hist =collections.defaultdict(list)

    while e <= args.FL_global_epochs:
        torch.cuda.empty_cache() 
        # random select
        # round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
        # round_malicious = round_users[round_users < n_attackers]
        # round_benign = round_users[round_users >= n_attackers]

        all_clients = np.arange(args.nClients)
        malicious_clients = np.random.choice(all_clients, n_attackers, replace=False)
        
        # Select clients for the round
        round_users = np.random.choice(all_clients, args.round_nclients, replace=False)

        num_round_malicious = int(args.round_nclients * args.at_fractions)
        round_malicious = np.random.choice(round_users, num_round_malicious, replace=False)
        round_benign = np.setdiff1d(round_users, round_malicious) 
            
        user_updates=collections.defaultdict(list)
        rs=collections.defaultdict(list)
        ########################################benign Client Learning#########################################
        m_c=collections.defaultdict(list)
        for n, m in FLmodel.named_modules():
            if hasattr(m, "scores"):
                m_c[str(n)]=0
            
        for kk in round_benign:
            mp = copy.deepcopy(FLmodel)
            optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
            
            scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
            for epoch in range(args.local_epochs):
                train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
                scheduler.step()
            
            for n, m in mp.named_modules():
                if hasattr(m, "scores"):
                    
                    rank=Find_rank(m.scores.detach().clone())
                    user_updates[str(n)]=rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank[None,:]), 0)
                    del rank
                    
        del optimizer, mp, scheduler
        ########################################malicious Client Learning######################################
        if len(round_malicious):
            for kk in np.random.choice(n_attackers, min(len(round_malicious), args.rand_mal_clients), replace=False): 
            
                torch.cuda.empty_cache()  
                mp = copy.deepcopy(FLmodel)
                optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
                scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
                for epoch in range(args.local_epochs):
                    train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
                    scheduler.step()
                for n, m in mp.named_modules():
                    if hasattr(m, "scores"):
                        rank=Find_rank(m.scores.detach().clone())
                        rs[str(n)]=rank[None,:] if len(rs[str(n)])==0 else torch.cat((rs[str(n)],rank[None,:]),0)                 
            mal_rank={}
            ########## VEM attack###########
            for n, m in mp.named_modules():
                if hasattr(m, "scores"):                        
                    if estimation =='alt':
                        mal_rank=VEM.optimize_alt(args.round_nclients,rs[str(n)],args.sparsity,len(round_malicious),args.device,args.lr_vem,args.nep,args.max_t,args.temp,args.iteration,args.noise)                     
                    elif estimation=='his':                          
                        if e>0: 
                            mal_rank=VEM.optimize_his(args.round_nclients,rs[str(n)],Rg_hist[str(n)], Rm_hist[str(n)],args.sparsity,len(round_malicious),args.device,args.lr_vem,args.nep,args.max_t,args.temp,args.iteration,args.noise)                              
                        else:
                            mal_rank=VEM.optimize_alt(args.round_nclients,rs[str(n)],args.sparsity,len(round_malicious),args.device,args.lr_vem,args.nep,args.max_t,args.temp,args.iteration,args.noise)  
                        Rm_hist[str(n)]=mal_rank
                    user_updates[str(n)]=mal_rank if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], mal_rank), 0)
            del mal_rank,mp

        ########################################Server AGR#########################################

        if args.agr=='Eud':  # you can change to others
            selected_user_updates=defense.Euclidean(FLmodel, user_updates,int(0.2*len(round_users)))
        elif args.agr=='Krum':
            selected_user_updates=defense.Krum(FLmodel, user_updates,int(0.2*len(round_users)))
        else:
            selected_user_updates=user_updates
        
        FRL_Vote(FLmodel, selected_user_updates, initial_scores)
        if estimation=='his':
            for n, m in FLmodel.named_modules():
                if hasattr(m, "scores"):
                    rg = Find_rank(m.scores.detach().clone())  # [n_edges]
                    Rg_hist[str(n)]=rg        # store per-round ranking

        del user_updates
        if (e+1)%1==0:
            t_loss, t_acc = test(te_loader, FLmodel, criterion, args.device) 
            if t_acc>t_best_acc:
                t_best_acc=t_acc

            sss='e %d | malicious users: %d | test acc %.4f test loss %.6f best test_acc %.4f' % (e, len(round_malicious), t_acc, t_loss, t_best_acc)
            print (sss)

            with (args.run_base_dir / "output.txt").open("a") as f:
                f.write("\n"+str(sss))

        e+=1

def FRL_other_attacks(tr_loaders,te_loader):
    # run = wandb.init()
    m_r=wandb.config.m_r

    print ("#########Federated Learning using Rankings############")
    args.conv_type = 'MaskConv'
    args.conv_init = 'signed_constant'
    args.bn_type="NonAffineNoStatsBN"    
    
    n_attackers = int(args.nClients * m_r)
    sss = "fraction of maliciou clients: %.2f | total number of malicious clients: %d"%(m_r,
                                                                                        n_attackers)
    print (sss)
    with (args.run_base_dir / "output.txt").open("a") as f:
        f.write("\n"+str(sss))
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    FLmodel = getattr(models, args.model)().to(args.device)
    
    initial_scores={}
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            initial_scores[str(n)]=m.scores.detach().clone().flatten().sort()[0]
    
    e=0
    t_best_acc=0
    while e <= args.FL_global_epochs:
        torch.cuda.empty_cache() 
        # round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
        # round_malicious = round_users[round_users < n_attackers]
        # round_benign = round_users[round_users >= n_attackers]
        # while len(round_malicious)>=args.round_nclients/2:
        #     round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
        #     round_malicious = round_users[round_users < n_attackers]
        #     round_benign = round_users[round_users >= n_attackers]
        all_clients = np.arange(args.nClients)
        malicious_clients = np.random.choice(all_clients, n_attackers, replace=False)
        
        # Select clients for the round
        round_users = np.random.choice(all_clients, args.round_nclients, replace=False)

        num_round_malicious = int(args.round_nclients * m_r)
        # Ensure exactly num_round_malicious malicious clients
        round_malicious = np.random.choice(round_users, num_round_malicious, replace=False)
        round_benign = np.setdiff1d(round_users, round_malicious) 
       
        user_updates=collections.defaultdict(list)
        scores=collections.defaultdict(list)
        ########################################benign Client Learning#########################################
        m_c=collections.defaultdict(list)
        for n, m in FLmodel.named_modules():
            if hasattr(m, "scores"):
                m_c[str(n)]=0
        for kk in round_benign:
            mp = copy.deepcopy(FLmodel)
            optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
            
            scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
            for epoch in range(args.local_epochs):
                train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
                scheduler.step()

            for n, m in mp.named_modules():
                if hasattr(m, "scores"):
                    rank=Find_rank(m.scores.detach().clone())
                    score=m.scores.flatten()
                        ########### pass m benign scores to attacker#############
                    if m_c[str(n)]<len(round_benign):
                        scores[str(n)]=score[None,:] if len(scores[str(n)])==0 else torch.cat((scores[str(n)],score[None,:]),0)       # get the score of all benign users
                        m_c[str(n)]=m_c[str(n)]+1
                    user_updates[str(n)]=rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank[None,:]), 0)
                    del rank
            del optimizer, mp, scheduler
        # ########################################malicious Client Learning old######################################
        attack=wandb.config.attacks
        if attack=='noise':
            if len(round_malicious):
                for kk in np.random.choice(n_attackers, min(len(round_malicious), args.rand_mal_clients), replace=False):
                    torch.cuda.empty_cache()  
                    mp = copy.deepcopy(FLmodel)
                    optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
                    scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
                    for n, m in mp.named_modules():
                        if hasattr(m, "scores"):                           
                            mal_scores=other_attacks.noise(scores[str(n)],args.device)                                         
                            rank=Find_rank_attack(mal_scores.detach().clone())   # get the opsite sign of ig score                           
                            user_updates[str(n)]=rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank[None,:]), 0)
                            del rank
                del optimizer, mp, scheduler
        ########################################malicious Client Learning new-all the same#####################################
        else:
            if len(round_malicious):
                # for kk in np.random.choice(n_attackers, min(len(round_malicious), args.rand_mal_clients), replace=False):
                torch.cuda.empty_cache()  
                mp = copy.deepcopy(FLmodel)
                optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
                scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
                for n, m in mp.named_modules():
                    if hasattr(m, "scores"):
                        if attack=='min_sum':
                            mal_scores=other_attacks.min_sum(scores[str(n)],args.device)
                        elif attack =='min_max':
                            mal_scores=other_attacks.min_max(scores[str(n)],args.device)
                        elif attack =='grad_ascent':
                            mal_scores=-m.scores.flatten()
                            
                        rank=Find_rank_attack(mal_scores.detach().clone())   # get the opsite sign of ig score
                        for kk in np.random.choice(n_attackers, min(len(round_malicious), args.rand_mal_clients), replace=False):
                            user_updates[str(n)]=rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank[None,:]), 0)
                        del rank
                del optimizer, mp, scheduler
        ########################################Server AGR#########################################
        if wandb.config.defense=='foolsgold':
            selected_user_updates=defense.foolsgold(FLmodel, user_updates,args.device,initial_scores)
  
        else:
            if wandb.config.defense=='cosine':
                selected_user_updates=defense.cosine(FLmodel, user_updates,int(0.2*len(round_users)))

            elif wandb.config.defense=='Eud':
                selected_user_updates=defense.Euclidean(FLmodel, user_updates,int(0.2*len(round_users)))

            elif wandb.config.defense=='Krum':
                selected_user_updates=defense.Krum(FLmodel, user_updates,int(0.2*len(round_users)))
            elif wandb.config.defense=='FABA':
                selected_user_updates=defense.FABA(FLmodel, user_updates,int(0.2*len(round_users)))
            elif wandb.config.defense=='DnC':
                selected_user_updates=defense.DnC(FLmodel, user_updates,int(0.2*len(round_users)))
           
            else:
                selected_user_updates=user_updates               
            
            FRL_Vote(FLmodel, selected_user_updates, initial_scores)

        del user_updates
        if (e+1)%1==0:
            t_loss, t_acc = test(te_loader, FLmodel, criterion, args.device) 
            if t_acc>t_best_acc:
                t_best_acc=t_acc

            sss='e %d | malicious users: %d | test acc %.4f test loss %.6f best test_acc %.4f' % (e, len(round_malicious), t_acc, t_loss, t_best_acc)
            # sss='e %d | malicious users: %d | test acc %.4f test loss %.6f best test_acc %.4f | max %.4f min %.4f' % (e, len(round_malicious), t_acc, t_loss, t_best_acc,max_val,min_val)
            print (sss)
            with (args.run_base_dir / "output.txt").open("a") as f:
                f.write("\n"+str(sss))
            wandb.log(
            {
                "val_acc": t_acc,
                "best_acc":t_best_acc,
                # "loss":t_loss,
                # "max":max_val,
                # "min":min_val,
                # "location":args.run_base_dir / "output.txt"

            })
        e+=1


def FRL_train(tr_loaders, te_loader):
    print ("#########Federated Learning using Rankings############")
    run = wandb.init()
    args.at_fractions=wandb.config.args.at_fractions
    args.conv_type = 'MaskConv'
    args.conv_init = 'signed_constant'
    args.bn_type="NonAffineNoStatsBN"    
    
    n_attackers = int(args.nClients * args.at_fractions)
    sss = "fraction of maliciou clients: %.2f | total number of malicious clients: %d"%(args.at_fractions,
                                                                                        n_attackers)
    print (sss)
    with (args.run_base_dir / "output.txt").open("a") as f:
        f.write("\n"+str(sss))
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    FLmodel = getattr(models, args.model)().to(args.device)
    
    initial_scores={}
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            initial_scores[str(n)]=m.scores.detach().clone().flatten().sort()[0]
    
    e=0
    t_best_acc=0
    
    
    print(args.args.lr_vem,args.lrdc, args.momentum, args.wd, args.local_epochs)
    while e <= args.FL_global_epochs:
        torch.cuda.empty_cache() 
        all_clients = np.arange(args.nClients)
        malicious_clients = np.random.choice(all_clients, n_attackers, replace=False)
        
        # Select clients for the round
        round_users = np.random.choice(all_clients, args.round_nclients, replace=False)

        num_round_malicious = int(args.round_nclients * args.at_fractions)
        # Ensure exactly nuargs.at_fractionsound_malicious malicious clients
        round_malicious = np.random.choice(round_users, num_round_malicious, replace=False)
        round_benign = np.setdiff1d(round_users, round_malicious) 

        # round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
        # round_malicious = round_users[round_users < n_attackers]
        # round_benign = round_users[round_users >= n_attackers]
        # while len(round_malicious)>=args.round_nclients/2:
        #     round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
        #     round_malicious = round_users[round_users < n_attackers]
        #     round_benign = round_users[round_users >= n_attackers]
            
        user_updates=collections.defaultdict(list)
        ########################################benign Client Learning#########################################
        for kk in round_benign:
            mp = copy.deepcopy(FLmodel)
            optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
            # optimizer = optim.Adam([p for p in mp.parameters() if p.requires_grad], args.lr_vem=args.args.lr_vem)
            
            scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
            for epoch in range(args.local_epochs):
                train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
                scheduler.step()

            for n, m in mp.named_modules():
                    if hasattr(m, "scores"):
                        rank=Find_rank(m.scores.detach().clone())
                        user_updates[str(n)]=rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank[None,:]), 0)
                        del rank
            del optimizer, mp, scheduler
        ########################################malicious Client Learning######################################
        if len(round_malicious):
            sum_args_sorts_mal={}
            for kk in np.random.choice(n_attackers, min(n_attackers, args.rand_mal_clients), replace=False):
                torch.cuda.empty_cache()  
                mp = copy.deepcopy(FLmodel)
                optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
                scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
                for epoch in range(args.local_epochs):
                    train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
                    scheduler.step()

                for n, m in mp.named_modules():    
                    if hasattr(m, "scores"):
                        rank=Find_rank(m.scores.detach().clone())      # get the rank of current score
                        rank_arg=torch.sort(rank)[1]
                        if str(n) in sum_args_sorts_mal:
                            sum_args_sorts_mal[str(n)]+=rank_arg       # aggreate the ranking of malicious
                        else:
                            sum_args_sorts_mal[str(n)]=rank_arg
                        del rank, rank_arg
                del optimizer, mp, scheduler

            for n, m in FLmodel.named_modules():
                if hasattr(m, "scores"):
                    rank_mal_agr=torch.sort(sum_args_sorts_mal[str(n)], descending=True)[1]    # simply sort in descending order
                    for kk in round_malicious:
                        user_updates[str(n)]=rank_mal_agr[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank_mal_agr[None,:]), 0)
            del sum_args_sorts_mal
        ########################################Server AGR#########################################
   
        del user_updates
        if (e+1)%1==0:
            t_loss, t_acc = test(te_loader, FLmodel, criterion, args.device) 
            if t_acc>t_best_acc:
                t_best_acc=t_acc

            sss='e %d | malicious users: %d | test acc %.4f test loss %.6f best test_acc %.4f' % (e, len(round_malicious), t_acc, t_loss, t_best_acc)
            print (sss)
            with (args.run_base_dir / "output.txt").open("a") as f:
                f.write("\n"+str(sss))
            
           
        e+=1

  
        
  
        
     