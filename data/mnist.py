from args import args
from torchvision import datasets, transforms
import torchvision
from data.Dirichlet_noniid import *
from data.iid import *
import wandb
from torch.utils.data import DataLoader, random_split

class MNIST:
    def __init__(self):
        
        args.output_size = 10
        
        Mytransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        train_dataset = datasets.MNIST(root=args.data_loc, train=True, download=True, transform=Mytransform)

        test_dataset = datasets.MNIST(root=args.data_loc, train=False, download=True, transform=Mytransform)

        if args.non_iid_degree=='iid':
            tr_per_participant_list = sample_iid_train_data(train_dataset, args.nClients)
        else:
            tr_per_participant_list, tr_diversity = sample_dirichlet_train_data_train(train_dataset, args.nClients, alpha=args.non_iid_degree, force=False)
     
        
        self.tr_loaders = []
        tr_count = 0
        for pos, indices in tr_per_participant_list.items():
            if len(indices)==1 or len(indices)==0:
                print (pos)
            tr_count += len(indices)
            batch_size = args.batch_size
            self.tr_loaders.append(get_train(train_dataset, indices, args.batch_size))
      
        if args.FL_type =="FRL_fang" or args.FL_type =="FRL_defense_Fang"or args.FL_type =='other_attacks_Fang'or args.FL_type =='FRL_label_flip_fang'or args.FL_type =='Reverse_mid_val'or args.FL_type =='other_attacks_agnostic_val'or args.FL_type =='FRL_train_agnostic_val': 
            
            validation_size = 100 
            test_size = len(test_dataset) - validation_size

            validation_dataset, test_dataset = random_split(test_dataset, [validation_size, test_size])
            self.validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.test_batch_size, shuffle=False)
            self.te_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
        else:
            self.te_loader= torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)


    def get_tr_loaders(self):
        return self.tr_loaders
    
    def get_te_loader(self):
        return self.te_loader
    
    def get_val_loader(self):
        return self.validation_loader