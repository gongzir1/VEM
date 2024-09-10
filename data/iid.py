from args import args
import pickle
from collections import defaultdict
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

def get_train(dataset, indices, batch_size=args.batch_size, shuffle=True):
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))
    
    return train_loader

def sample_iid_train_data(train_dataset, no_participants, force=False):
    file_add = '%s_train_iid_n%d.pkl' % (args.set, no_participants)
    if not os.path.exists(file_add) or force:
        print('generating IID participant indices')

        # Total number of data points
        num_data_points = len(train_dataset)

        # Number of data points per participant
        data_per_participant = num_data_points // no_participants

        # Shuffle indices for random distribution
        all_indices = list(range(num_data_points))
        random.shuffle(all_indices)

        tr_per_participant_list = defaultdict(list)

        # Distribute data indices evenly to participants
        for user in range(no_participants):
            start_index = user * data_per_participant
            end_index = (user + 1) * data_per_participant
            tr_per_participant_list[user].extend(all_indices[start_index:end_index])

        # In case of any remaining data points, distribute them to participants
        remaining_indices = all_indices[no_participants * data_per_participant:]
        for i, idx in enumerate(remaining_indices):
            tr_per_participant_list[i % no_participants].append(idx)

        with open(file_add, 'wb') as f:
            pickle.dump(tr_per_participant_list, f)
    else:
        tr_per_participant_list = pickle.load(open(file_add, 'rb'))

    return tr_per_participant_list

# def plot_distribution(tr_per_participant_list, no_participants):
#     counts = [len(tr_per_participant_list[i]) for i in range(no_participants)]
#     plt.figure(figsize=(10, 6))
#     plt.bar(range(no_participants), counts)
#     plt.xlabel('Participant')
#     plt.ylabel('Number of Data Points')
#     plt.title('Data Distribution Among Participants (IID)')
#     plt.show()

