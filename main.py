import os
from args import args
import random
import numpy as np
import pathlib
import torch

import data
from FL_train import *
from args import *



if args.FL_type =='other_attacks':
    config={
    "method": "grid",
    "metric":{
        "goal": "minimize", 
        "name": "t_acc"
        },
    "parameters":{
    'k':{"values": [0.1]},
    'm_r':{"values": [0.2]},
    'non_iid':{"values": ['iid']},
    'attacks':{"values": ['grad_ascent','min_max','min_sum','noise']},
    # 'attacks':{"values": ['noise']},
    'defense':{"values": ['my_defense_adaptive']},
    # 'k_a':{"values": [0.5,1]},
    # 'maxt':{"values": [1000]},
    # 'mint':{"values": [5,10,15,20,25,30]},
    # 'mint':{"values": [10,15,5]},
    # 'attacks':{"values": ['lable']},
      },
    }
elif args.FL_type =='FRP_defense':
       config={
    "method": "grid",
    "metric":{
        "goal": "minimize", 
        "name": "t_acc"
        },
    "parameters":{
    'k':{"values": [0.1]},
    'm_r':{"values": [0.2]},
    'non_iid':{"values": [1]},
    'attacks':{"values": ['rank-reverse']},
    'defense':{"values": ['FRL']},
    },
    }
elif args.FL_type =='FRL_defense_Fang':
       config={
    "method": "grid",
    "metric":{
        "goal": "minimize", 
        "name": "t_acc"
        },
    "parameters":{
    'k':{"values": [0.1]},
    'm_r':{"values": [0.2]},
    'non_iid':{"values": [0.1,0.5,'iid']},
    'attacks':{"values": ['rank-reverse']},
    'defense':{"values": ['fl_trust']},
    },
    }
elif args.FL_type =='FRL_fang':
       config={
    "method": "grid",
    "metric":{
        "goal": "minimize", 
        "name": "t_acc"
        },
    "parameters":{
    "lr":{"values": [0.01]},
    "nep":{"values": [40]},
    "max_t":{"values": [2000]},
    "iteration":{"values": [40]},
    "temp":{"values": [0.0001]},
    "noise":{"values": [1]},
    'k':{"values": [0.1]},
    'm_r':{"values": [0.2]},
    'non_iid':{"values": [0.1,0.3,0.5,0.7,0.9]},
    'attacks':{"values": ['my_attack']},
    # 'defense':{"values": ['FRL_fang']},
    'defense':{"values": ['fl_trust']},
    # 'mode':{"values": ['ERR','LFR','combined']},
    },
    }
elif args.FL_type =='my_attack_defense' or args.FL_type=='FRL_matrix_attack_defense_upper_bound' or args.FL_type=='FRL_matrix_attack_defense_forcasting' or args.FL_type=='compare_different_estimation':
       config={
    "method": "grid",
    "metric":{
        "goal": "minimize", 
        "name": "t_acc"
        },
    "parameters":{
    "lr":{"values": [0.1]},
    "nep":{"values": [60]},
    "max_t":{"values": [2500]},
    "iteration":{"values": [50]},
    "temp":{"values": [0.0001]},
    "noise":{"values": [1]},
    'k':{"values": [0.1]},
    'm_r':{"values": [0.2]},
    'non_iid':{"values": ['iid']},
    'attacks':{"values": ['my_attack']},
    'defense':{"values": ['FRL']}, 
    },
    }
elif args.FL_type =='Reverse_mid'or args.FL_type =='FRL_matrix_attack':
       config={
    "method": "grid",
    "metric":{
        "goal": "minimize", 
        "name": "t_acc"
        },
    "parameters":{
    "lr":{"values": [0.2]},
    "nep":{"values": [40,50,60]},
    "max_t":{"values": [2000]},
    "iteration":{"values": [40]},
    "temp":{"values": [0.001]},
    "noise":{"values": [1]},
    'k':{"values": [0.1]},
    'm_r':{"values": [0.2]},
    'non_iid':{"values": [1]},
    'attacks':{"values": ['my_attack_new']},
    'defense':{"values": ['FRL']},
    },
    }
else:
    config={
    "method": "grid",
    "metric":{
        "goal": "minimize", 
        "name": "t_acc"
        },
    "parameters":{
    'k':{"values": [0.5]},
    'm_r':{"values": [0.2]},
    'non_iid':{"values": ['iid']},
    },
    }
     
       


def main():
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
    # Make the a directory corresponding to this run for sval_loaderaving results, checkpoints etc.
    i = 0
    while True:
        run_base_dir = pathlib.Path(f"{args.log_dir}/"+args.set+args.FL_type+f"~try={str(i)}")

        if not run_base_dir.exists():
            os.makedirs(run_base_dir)
            args.name = args.name + f"~try={i}"
            break
        i += 1

    (run_base_dir / "output.txt").write_text(str(args))
    args.run_base_dir = run_base_dir

    print(f"=> Saving data in {run_base_dir}")
     
    #distribute the dataset
    print ("dataset to use is: ", args.set)
    print ("number of FL clients: ", args.nClients)
    # print ("non-iid degree data distribution: ", args.non_iid_degree)
    print ("non-iid degree data distribution: ", args.non_iid_degree)
    print ("batch size is : ", args.batch_size)
    print ("test batch size is: ", args.test_batch_size)
    
    data_distributer = getattr(data, args.set)()
    if args.FL_type == "FRL_defense_Fang" or args.FL_type =="FRL_fang" or args.FL_type =='other_attacks_Fang'or args.FL_type == 'FRL_label_flip_fang' or args.FL_type =='Reverse_mid_val'or args.FL_type =='other_attacks_agnostic_val' or args.FL_type =='FRL_train_agnostic_val':  
        tr_loaders = data_distributer.get_tr_loaders()    # len=10000 list
        te_loader = data_distributer.get_te_loader()
        val_loader = data_distributer.get_val_loader()
    else:
        tr_loaders = data_distributer.get_tr_loaders()
        te_loader = data_distributer.get_te_loader()
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()
    print ("use_cuda: ", use_cuda) 
    
    #Federated Learning
    print ("type of FL: ", args.FL_type)
    if args.FL_type == "FRL":
        FRL_train(tr_loaders, te_loader)
    elif args.FL_type =="VEM":
        FRL_VEM(tr_loaders, te_loader)



    
if __name__ == "__main__":
    main()
