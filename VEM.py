import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function
import permutation_ops
from scipy.optimize import linear_sum_assignment
 
import torch

def calculate_threshold_historical(r_s,Rg_hist, Rm_hist, u, m, k, max_t):

    w_his=torch.sort(Rm_hist,1)[1]  # repution for m updates
    w_m_his=torch.sum(w_his,0)   # aggreated reputation
    wg=torch.sort(Rg_hist)[1]

    wg_u=u*wg
    w_benign=wg_u-w_m_his
    sorted_w=torch.sort(w_benign)[0]

    w=torch.sort(r_s,1)[1]  # repution for m updates
    w_m=torch.sum(w,0) 
    
    n=w.size(1)
    w_k=sorted_w[int((1-k)*n)]

    w_s=w_k-m*(n-1)
    w_s2=w_k+m*(n-1)

    w=torch.sort(r_s,1)[1]  # repution for m updates
    w_m=torch.sum(w,0)   
    

    # find threshold v 
    # t1=torch.nonzero(torch.lt(sorted_w,w_s))[-1]
    # t2=torch.nonzero(torch.gt(sorted_w,w_s2))[0]
    # handle out of range isseu
    smaller_indices = torch.nonzero(torch.lt(sorted_w, w_s))
    greater_indices=torch.nonzero(torch.gt(sorted_w,w_s2))

    # handle the case where no elements are smaller than the threshold
    t1 = smaller_indices[-1] if smaller_indices.numel() > 0 else torch.tensor(0)
    t2 = greater_indices[0] if greater_indices.numel()>0 else torch.tensor(n)

    # solve out of memory issue
    if t2 - t1 > max_t:
        t1 = torch.tensor(int(n * (1 - k)) - max_t//2, dtype=torch.int32) if torch.tensor(int(n * (1 - k)) - max_t//2, dtype=torch.int32)>0 else torch.tensor(0)
        t2 = torch.tensor(int(n * (1 - k)) + max_t//2, dtype=torch.int32) if torch.tensor(int(n * (1 - k)) + max_t//2, dtype=torch.int32)<n else torch.tensor(n)

    # w_t1=sorted_w[t1]
    # w_t2=sorted_w[t2]

    
    return t1,t2,w,w_m
    

def calculate_threshold_alternative(r_s,u,m,k,max_t):

    w=torch.sort(r_s,1)[1]  # repution for m updates
    w_m=torch.sum(w,0)   # aggreated reputation e_1, e_2, ... e_n
    
    # get s_all
    w_benign=w_m*(int(u/m)-1)
    sorted_w=torch.sort(w_benign)[0]
    sorted_edges=torch.sort(w_m)[1]
    # get the w
    n=w.size(1)
    w_k=sorted_w[int((1-k)*n)]

    w_s=w_k-m*(n-1)
    w_s2=w_k+m*(n-1)

    # find threshold v 
    # t1=torch.nonzero(torch.lt(sorted_w,w_s))[-1]
    # t2=torch.nonzero(torch.gt(sorted_w,w_s2))[0]
    # handle out of range isseu
    smaller_indices = torch.nonzero(torch.lt(sorted_w, w_s))
    greater_indices=torch.nonzero(torch.gt(sorted_w,w_s2))

    # handle the case where no elements are smaller than the threshold
    t1 = smaller_indices[-1] if smaller_indices.numel() > 0 else torch.tensor(0)
    t2 = greater_indices[0] if greater_indices.numel()>0 else torch.tensor(n)

    # solve out of memory issue

    if t2 - t1 > max_t:
        t1 = torch.tensor(int(n * (1 - k)) - max_t//2, dtype=torch.int32) if torch.tensor(int(n * (1 - k)) - max_t//2, dtype=torch.int32)>0 else torch.tensor(0)
        t2 = torch.tensor(int(n * (1 - k)) + max_t//2, dtype=torch.int32) if torch.tensor(int(n * (1 - k)) + max_t//2, dtype=torch.int32)<n else torch.tensor(n)
    
    return t1,t2,w,w_m


def Hungarian_algorithm(matrix):
    """
    Convert an approximate permutation matrix to an exact permutation matrix using the Hungarian algorithm.
    
    Args:
        matrix: Tensor of shape [batch_size, n, n]
    
    Returns:
        exact_matrix: Tensor of shape [batch_size, n, n] with one-hot permutation matrices.
    """
    batch_size, n, _ = matrix.size()
    exact_matrix = torch.zeros_like(matrix)
    
    for b in range(batch_size):
        cost_matrix = -matrix[b].detach().cpu().numpy()  # Convert to numpy and negate for maximization problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        exact_matrix[b, row_ind, col_ind] = 1
    
    return exact_matrix

def optimize_alt(u,r_s,k,m,device,lr,nep,max_t,temp,iteration,noise):
    
    t1,t2,ranking,w_m=calculate_threshold_alternative(r_s,u,m,k,max_t) 
    
    sorted_reputation=torch.sort(w_m)

    sorted_reputation_sub=sorted_reputation[0][t1:t2].float() # get the aggreated repution for vunerable edges
    vunerable_edges=sorted_reputation[1][t1:t2]    # vunerable edges
    vunerable=ranking[:,vunerable_edges].float().unsqueeze(1)

    sorted_reputation_sub.requires_grad=True

    # create permutation matrix with size (t2-t1)*(t2-t1)
    E=torch.eye((t2-t1).item(),requires_grad=True).to(device)
    E_1=[E for _ in range (m)]
    E_sub=torch.stack(E_1)

    # gumble softmax optimise logits
    logits=torch.rand_like(E_sub)
    logits.requires_grad=True

    optim=torch.optim.Adam([logits],lr)
    for i in range(nep):
        # apply gumble softmax 

        E_sub_train,b=permutation_ops.my_gumbel_sinkhorn(logits,temp,noise_factor=noise,n_iters=iteration)

        rank_mal=torch.matmul(vunerable,E_sub_train)
        w_mal_agg=torch.sum(rank_mal,0).squeeze(0)   # get the aggreated reputation for vunerable edges

        f=torch.norm((sorted_reputation_sub-w_mal_agg),p=2)
        Loss=-f
        
        optim.zero_grad()
        Loss.backward()
        optim.step()
 
    E_final=Hungarian_algorithm(E_sub_train)
    mal_rank_vunberable_edges=((vunerable)@E_final).squeeze(1)
    ranking[:,vunerable_edges]=mal_rank_vunberable_edges.long()      #update w 
    
    return ranking

def optimize_his(u,r_s,Rg_hist, Rm_hist,k,m,device,lr,nep,max_t,temp,iteration,noise):
    
    t1,t2,ranking,w_m=calculate_threshold_historical(r_s,Rg_hist, Rm_hist, u, m, k, max_t)
    sorted_reputation=torch.sort(w_m)

    sorted_reputation_sub=sorted_reputation[0][t1:t2].float() # get the aggreated repution for vunerable edges
    vunerable_edges=sorted_reputation[1][t1:t2]    # vunerable edges
    vunerable=ranking[:,vunerable_edges].float().unsqueeze(1)

    sorted_reputation_sub.requires_grad=True

    # create permutation matrix with size (t2-t1)*(t2-t1)
    E=torch.eye((t2-t1).item(),requires_grad=True).to(device)
    E_1=[E for _ in range (m)]
    E_sub=torch.stack(E_1)

    # gumble softmax optimise logits
    logits=torch.rand_like(E_sub)
    logits.requires_grad=True

    optim=torch.optim.Adam([logits],lr)
    for i in range(nep):
        # apply gumble softmax 
        E_sub_train,b=permutation_ops.my_gumbel_sinkhorn(logits,temp,noise_factor=noise,n_iters=iteration)

        rank_mal=torch.matmul(vunerable,E_sub_train)
        w_mal_agg=torch.sum(rank_mal,0).squeeze(0)   # get the aggreated reputation for vunerable edges

        f=torch.norm((sorted_reputation_sub-w_mal_agg),p=2)
        Loss=-f
        
        optim.zero_grad()
        Loss.backward()
        optim.step()
 
    E_final=Hungarian_algorithm(E_sub_train)
    mal_rank_vunberable_edges=((vunerable)@E_final).squeeze(1)
    ranking[:,vunerable_edges]=mal_rank_vunberable_edges.long()      # update w 
    
    return ranking


