#!/usr/bin/env python
# coding: utf-8

# In[2]:


#packages


import numpy as np

import copy
import os 
import gc 
import pickle
import time
import matplotlib.pyplot as plt



import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.data import *
from src.models import *
from src.fedavg import *
from src.client import * 
from src.clustering import *
from src.utils import * 


from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


st = time.time()
args = args_parser()

# ---- DEVICE SETUP ----
if torch.cuda.is_available():
    # pick GPU 0 on your A100 box, or read from env / config
    args.gpu = 0                     # or set this earlier based on your setup
    torch.cuda.set_device(args.gpu)  # respects CUDA_VISIBLE_DEVICES
    args.device = torch.device(f"cuda:{args.gpu}")
    torch.backends.cudnn.benchmark = True  # speed up convs for fixed input sizes
    print(f"Using GPU: {args.device}")
else:
    args.gpu = -1
    args.device = torch.device("cpu")
    print("CUDA not available, using CPU")


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

#parameters for flag setup
args.local_view=True
# args.model='simple-cnn'
# args.dataset='cifar10'
# args.partition='flag-non-iid'
# args.num_users=100
# args.rounds=81
# args.frac=.2
# args.beta=1
r=1.5
print("Paramters : ",str(args))
path = args.savedir + args.alg + '/' + args.partition + '/' + args.dataset + '/'
mkdirs(path)

template = "Algorithm {}, Clients {}, Dataset {}, Model {}, Non-IID {}, Threshold {}, K {}, Linkage {}, LR {}, Ep {}, Rounds {}, bs {}, frac {}"

s = template.format(args.alg, args.num_users, args.dataset, args.model, args.partition, args.cluster_alpha, args.n_basis, args.linkage, args.lr, args.local_ep, args.rounds, args.local_ep, args.frac)


# In[3]:


##################################### Data partitioning section 
args.local_view = True
X_train, y_train, X_test, y_test, net_dataidx_map, net_dataidx_map_test, \
traindata_cls_counts, testdata_cls_counts = partition_data(args.dataset, 
args.datadir, args.logdir, args.partition, args.num_users, beta=args.beta, local_view=args.local_view)


# In[4]:


pin_mem = (args.device.type == "cuda")
num_workers = 4  # A good starting point; you can try 8–16

train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(
    args.dataset,
    args.datadir,
    args.local_bs,
    32,
    dataidxs=None,
    noise_level=0,
    net_id=None,
    total=args.num_users,
    dataidxs_test=None,
    same_size=False,
    target_transform=None,
    rotation=0,
    num_workers=num_workers,
    pin_memory=pin_mem,
)

print("len train_ds_global:", len(train_ds_global))
print("len test_ds_global:", len(test_ds_global))


################################### build  single fe+cl model
def init_nets(args, dropout_p=0.5):

    users_model = []

    for net_i in range(-1, args.num_users):
        if args.model == "simple-cnn":
            if args.dataset in ("cifar10", "cinic10", "svhn"):
                #net = CNN_CIFAR.to(args.device) #changed delete
                # net = SimpleCNNLight(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).to(args.device)
                net = SimpleCNN(num_classes=10).to(args.device) 
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                net = SimpleCNNMNIST2().to(args.device)
               # net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10).to(args.device)
            elif args.dataset == 'cifar100':
                #net = SimpleCNNLight(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=100).to(args.device)
 
                net = SimpleCNN_3(input_dim=(16 * 3 * 5 * 5), hidden_dims=[120*3, 84*3], output_dim=100).to(args.device)
                #
        elif args.model =="simple-cnn-3":
            if args.dataset == 'cifar100': 
                net = SimpleCNN_3(input_dim=(16 * 3 * 5 * 5), hidden_dims=[120*3, 84*3], output_dim=100).to(args.device)
            if args.dataset == 'tinyimagenet':
                net = SimpleCNNTinyImagenet_3(input_dim=(16 * 3 * 13 * 13), hidden_dims=[120*3, 84*3], 
                                              output_dim=200).to(args.device)
        # elif args.model == "vgg-9":
        #     if args.dataset in ("mnist", 'femnist'):
        #         net = ModerateCNNMNIST().to(args.device)
        #     elif args.dataset in ("cifar10", "cinic10", "svhn"):
        #         # print("in moderate cnn")
        #         net = ModerateCNN().to(args.device)
        #     elif args.dataset == 'celeba':
        #         net = ModerateCNN(output_dim=2).to(args.device)
        elif args.model == 'resnet9': 
            if args.dataset == 'cifar100':
                net = SimpleCNN_3(input_dim=(16 * 3 * 5 * 5), hidden_dims=[120*3, 84*3], output_dim=100).to(args.device)
                #net = ResNet9(in_channels=3, num_classes=100)
            elif args.dataset == 'tinyimagenet': 
                net = ResNet9(in_channels=3, num_classes=200, dim=512*2*2)
        else:
            print("not supported yet")
            exit(1)
        if net_i == -1: #initializing the global model
            net_glob = copy.deepcopy(net) 
            
           
            initial_state_dict = copy.deepcopy(net_glob.state_dict())
            server_state_dict = copy.deepcopy(net_glob.state_dict())
            print("Global model device:", next(net_glob.parameters()).device)

            if args.load_initial:
                initial_state_dict = torch.load(args.load_initial)
                server_state_dict = torch.load(args.load_initial)
                net_glob.load_state_dict(initial_state_dict)
 
        else:
            users_model.append(copy.deepcopy(net))
            users_model[net_i].load_state_dict(initial_state_dict)

#     model_meta_data = []
#     layer_type = []
#     for (k, v) in nets[0].state_dict().items():
#         model_meta_data.append(v.shape)
#         layer_type.append(k)

    return users_model, net_glob, initial_state_dict, server_state_dict

print(f'MODEL: {args.model}, Dataset: {args.dataset}')

users_model, net_glob, initial_state_dict, server_state_dict = init_nets(args, dropout_p=0.5)

print(net_glob)

total = 0 
for name, param in net_glob.named_parameters():
    print(name, param.size())
    total += np.prod(param.size())
    #print(np.array(param.data.cpu().numpy().reshape([-1])))
    #print(isinstance(param.data.cpu().numpy(), np.array))
print(total)

################################# Initializing Clients 
traindata_cls_ratio = {}
pretrain_data1=[] #changed
pretrain_data2=[] #changed

budget = 50
for i in range(args.num_users):
    total_sum = sum(list(traindata_cls_counts[i].values()))
    base = 1/len(list(traindata_cls_counts[i].values()))
    temp_ratio = {}
    for k in traindata_cls_counts[i].keys():
        ss = traindata_cls_counts[i][k]/total_sum
        temp_ratio[k] = (traindata_cls_counts[i][k]/total_sum)
        # if ss >= (base + 0.05): 
        #     temp_ratio[k] = traindata_cls_counts[i][k]  #if any ratio excceds the base, use the 
            #actual value instead of ratios
            
    sub_sum = sum(list(temp_ratio.values()))
    for k in temp_ratio.keys():
        temp_ratio[k] = (temp_ratio[k]/sub_sum)*budget      #normalize the ratios to budget
    
    round_ratio = round_to(list(temp_ratio.values()), budget) #round ratios so they are int
    cnt = 0 
    for k in temp_ratio.keys():
        temp_ratio[k] = round_ratio[cnt]
        cnt+=1
        
    traindata_cls_ratio[i] = temp_ratio  
    
clients = []
U_clients = []

K = args.n_basis
#K = 5
U_dict={}
for idx in range(args.num_users):
    
    dataidxs = net_dataidx_map[idx]
    if net_dataidx_map_test is None:
        dataidx_test = None 
    else:
        dataidxs_test = net_dataidx_map_test[idx]

    #print(f'Initializing Client {idx}')

    noise_level = args.noise
    if idx == args.num_users - 1:
        noise_level = 0

    if args.noise_type == 'space':
        train_dl_local, test_dl_local, train_ds_local, test_ds_local, pre_train_dl, validation_dl = get_dataloader(
            args.dataset,
            args.datadir,
            args.local_bs,
            32,
            dataidxs,
            noise_level,
            idx,
            args.num_users - 1,
            dataidxs_test=dataidxs_test,
            num_workers=num_workers,
            pin_memory=pin_mem,
        )
    else:
        noise_level = args.noise / (args.num_users - 1) * idx
        train_dl_local, test_dl_local, train_ds_local, test_ds_local, pre_train_dl, validation_dl = get_dataloader(
            args.dataset,
            args.datadir,
            args.local_bs,
            32,
            dataidxs,
            noise_level,
            dataidxs_test=dataidxs_test,
            num_workers=num_workers,
            pin_memory=pin_mem,
        )

    idxs_local = np.arange(len(train_ds_local.data))
    labels_local = np.array(train_ds_local.target)
    # Sort Labels Train 
    idxs_labels_local = np.vstack((idxs_local, labels_local))
    idxs_labels_local = idxs_labels_local[:, idxs_labels_local[1, :].argsort()]
    idxs_local = idxs_labels_local[0, :]
    labels_local = idxs_labels_local[1, :]
    
    uni_labels, cnt_labels = np.unique(labels_local, return_counts=True)
    
    print(f'Labels: {uni_labels}, Counts: {cnt_labels}')
    
    nlabels = len(uni_labels)
    cnt = 0
    U_temp = []
    D_temp = {}
    for j in range(nlabels):
        local_ds1 = train_ds_local.data[idxs_local[cnt:cnt+cnt_labels[j]]]
        x=local_ds1.shape[0]
        local_ds1_copy = copy.deepcopy(local_ds1)
        #print(train_ds_local[idxs_local[cnt]][0])
        #show_tensor_image(train_ds_local[idxs_local[cnt]][0])
        local_ds1 = local_ds1.reshape(cnt_labels[j], -1)
         
        local_ds1 = local_ds1.T
        if type(train_ds_local.target[idxs_local[cnt:cnt+cnt_labels[j]]]) == torch.Tensor:
            label1 = list(set(train_ds_local.target[idxs_local[cnt:cnt+cnt_labels[j]]].numpy()))
        else:
            label1 = list(set(train_ds_local.target[idxs_local[cnt:cnt+cnt_labels[j]]]))
        assert len(label1) == 1
        
        #print(f'Label {j} : {label1}')
        
        
        #pretrain_data1.extend(select_representative_images(copy.deepcopy(local_ds1_copy),cnt_labels[j], 8, 30, label1[0])) #changed, delete, para=2nd
        # if local_ds1_copy.shape[0]>1:
        #     pretrain_data2.extend(select_representative_images_with_svd(copy.deepcopy(local_ds1_copy), x, label1[0]))
        if args.partition == 'noniid-labeldir': 
            #print('Dir partition')
            if label1 in list(traindata_cls_ratio[idx].keys()): 
                K = traindata_cls_ratio[idx][label1[0]]
            else: 
                K = min(args.n_basis,x)    #changed, K = args.n_basis
        if K > 0:
            u1_temp, sh1_temp, vh1_temp = np.linalg.svd(local_ds1, full_matrices=False)
            u1_temp=u1_temp/np.linalg.norm(u1_temp, ord=2, axis=0)
            U_temp.append(u1_temp[:, 0:K])
            D_temp[label1[0]] = u1_temp[:, 0:K] #changed
            
        cnt+=cnt_labels[j]
        
    #U_temp = [u1_temp[:, 0:K], u2_temp[:, 0:K]]
    U_clients.append(copy.deepcopy(np.hstack(U_temp)))
    U_dict[idx]= D_temp
    
    print(f'Shape of U: {U_clients[-1].shape}')
    
    clients.append(Client_ClusterFL(idx, copy.deepcopy(users_model[idx]), args.local_bs, args.local_ep, 
               args.lr, args.momentum, args.device, train_dl_local, test_dl_local,  pre_train_dl, validation_dl))

############### Getting gradient similarity matrix
clients_backup = [copy.deepcopy(client) for client in clients] # Make a backup of clients' models

# --- Added: build mask using dimension of one flattened gradient
#    Used to pick random indices for all clients graidients, a small subset for compression
#    Assosscation: compress_gradient_with_mask, generate_sparsity_mask, Modified pairwise_angles
#    Assosscation: Modified: flatten + sparsify in gradient similarity
compression_ratio = 50            # e.g. keep 1/100 of coordinates
first_diff = clients[0].get_W()
first_flat = flatten(first_diff).detach().cpu().numpy()
D = first_flat.shape[0]
mask = generate_sparsity_mask(D, compression_ratio)
# --- End of Added

list_of_dW=[]
#w_glob = copy.deepcopy(initial_state_dict)
for iteration in range(2): #run the model for 20 iterations before gettin gradients edited
    idxs_users = np.arange(args.num_users) #get all users id
    print(f'###### Gradient data ROUND {iteration+1} ######')
    list_of_dW.clear()
    #get global state and train
    for idx in idxs_users:
        #clients[idx].set_state_dict(copy.deepcopy(w_glob))
        oldW = clients[idx].get_W()
        loss = clients[idx].train(is_print=False)
        newW = clients[idx].get_W()
        difference_W = copy.deepcopy(oldW)
        subtract_(target=difference_W, minuend=newW, subtrahend=oldW)
        
        # --- Modified: flatten + sparsify
        flat = flatten(difference_W).detach().cpu().numpy()
        sparse = compress_gradient_with_mask(flat, mask)
        list_of_dW.append(sparse)
        # --- End of Modified
        
        # list_of_dW.append(difference_W) 
        #-uncomment incase remove the  "Modified: flatten + sparsify",
        #i.e, remove gradient compression
        
       
    
Grad_similarites = pairwise_angles(list_of_dW) 
clients = [copy.deepcopy(client) for client in clients_backup] # Reset clients using backup

######## Getting class-wise weighted data similarity matrix

def compute_w(fc1, fc2):
    if fc1 == 0 or fc2 == 0:
        return None  # Skip positions where fc1 or fc2 is 0
    
    max_val = max(fc1, fc2)
    min_val = min(fc1, fc2)
    
    w = max_val/min_val
    return w

def normalize_weights(weights, r):
    # Flatten the weights matrix while excluding None values
    valid_weights = [w for w in weights.flatten() if w is not None]
    
    min_weight = min(valid_weights)
    max_weight = max(valid_weights)
    
    # Normalize weights to the range [1, 1 + r]
    normalized_weights = np.full_like(weights, None, dtype=np.float64)
    for idx in np.ndindex(weights.shape):
        if weights[idx] is not None:
            normalized_weights[idx] = (weights[idx] - min_weight) / (max_weight - min_weight) * r + 1
   
    return normalized_weights

def calculating_weighted_adjacency(nclients, U_dict, labels, traindata_cls_counts, r):
    sim_mat = np.zeros([nclients, nclients])
    weights = np.full([nclients, nclients, labels], None)  # 3D array to store weights, initialized with None
    
    # Compute and store weights
    for idx1 in range(nclients):
        for idx2 in range(nclients):
            if idx1 == idx2:
                continue
            for l in range(labels):
                if l in traindata_cls_counts[idx1] and l in traindata_cls_counts[idx2]:
                    f1 = traindata_cls_counts[idx1][l]
                    f2 = traindata_cls_counts[idx2][l]
                    weights[idx1, idx2, l] = compute_w(f1, f2)  # Parameter b
    
    # Normalize the weights matrix
    normalized_weights = normalize_weights(weights, r)
    
    # Compute the similarity matrix using normalized weights
    for idx1 in range(nclients):
        for idx2 in range(nclients):
            if idx1 == idx2: 
                sim_mat[idx1, idx2] = 0
                continue
            
            angles = []
            for l in range(labels):
                if l in U_dict[idx1] and l in U_dict[idx2]:
                    U1 = copy.deepcopy(U_dict[idx1][l])
                    U2 = copy.deepcopy(U_dict[idx2][l])
                    mul = np.clip(U1.T @ U2, a_min=-1.0, a_max=1.0)
                    angle = np.min(np.arccos(mul)) * 180 / np.pi
                    weight = normalized_weights[idx1, idx2, l]
                    if weight is not None:
                        angle = min(angle * weight, 180)
                    angles.append(angle)
                elif l not in traindata_cls_counts[idx1] and l not in traindata_cls_counts[idx2]:
                    angles.append(0)
                else:
                    angles.append(180)
            
            sim_mat[idx1, idx2] = sum(angles) / len(angles)
    
    return sim_mat

v=calculating_weighted_adjacency(args.num_users, U_dict, 10,traindata_cls_counts,.5) #label hardcode
G= normalize_matrix(Grad_similarites)
v= normalize_matrix(v)


# In[5]:


#Fusion-gate: combing data and gradient
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# 1.  MLP that outputs an n-dimensional sigmoid-gated vector w ∈ (0,1)^n
# ----------------------------------------------------------------------
class RowGateMLP(nn.Module):
    def __init__(self, n: int, hidden: int = 128):
        """
        Args:
            n (int)     : number of clients  (size of similarity matrices)
            hidden (int): hidden layer width
        """
        super().__init__()
        d_in = 2 * n * n            # vec(G) ‖ vec(P)
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n),    # logits for each client
            nn.Sigmoid()             # element-wise → w ∈ (0,1)^n
        )

    def forward(self, G: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        flat = torch.cat([G.flatten(), P.flatten()], dim=0)   # (2n²,)
        return self.net(flat)                                 # (n,)

# ----------------------------------------------------------------------
# 2.  Fusion  (upper triangle uses row-gate, mirrored for symmetry)
# ----------------------------------------------------------------------
def fuse_similarity(G: torch.Tensor,
                    P: torch.Tensor,
                    gate_net: RowGateMLP):
    """
    Returns
        A : symmetric fused similarity  (n×n)
        w : vector of row gates         (n,)
    Formula (for i<j):
        A_ij = w_i G_ij + (1-w_i) P_ij ;  A_ji = A_ij ;  A_ii = 0
    """
    w = gate_net(G, P)                       # (n,)
    n = w.size(0)
    w_col = w.view(n, 1)                     # (n,1) for broadcasting

    mix = w_col * G + (1 - w_col) * P        # apply gate row-wise
    upper = torch.triu(mix, diagonal=1)      # keep strict upper triangle
    A = upper + upper.T                      # mirror → symmetric, zero diag

    return A, w

# ----------------------------------------------------------------------
# 3.  Row-softmax entropy loss  (no Laplacian term)
# ----------------------------------------------------------------------
def entropy_loss(A: torch.Tensor) -> torch.Tensor:
    n = A.size(0)
    soft = F.softmax(A, dim=1)
    logt = torch.clamp(soft, 1e-12, 1).log()
    return -(soft * logt).sum() / n          # minimise → sharper rows

# ----------------------------------------------------------------------
# 4.  Training example  (G and v are provided 100×100 NumPy or Torch)
# ----------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

def to_tensor(mat):
    if torch.is_tensor(mat):
        return mat.clone().to(device)        # duplicate to avoid in-place edits
    return torch.as_tensor(mat, dtype=torch.float32, device=device)

G_t = to_tensor(G)          # gradient similarity (100×100)
V_t = to_tensor(v)          # data      similarity (100×100)

n = G_t.size(0)
gate_net = RowGateMLP(n).to(device)
optimizer = torch.optim.Adam(gate_net.parameters(), lr=1e-3)

for step in range(50):
    optimizer.zero_grad()
    A, w = fuse_similarity(G_t, V_t, gate_net)
    loss = entropy_loss(A)
    loss.backward()
    optimizer.step()

print("Learned row-gates w (first 10 shown):", w[:90].cpu().detach().numpy())



# In[6]:


# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import copy
# from scipy.cluster.hierarchy import linkage, fcluster
# from scipy.spatial.distance import squareform


# # 2. Loss function (unchanged)
# def clustering_loss_with_tiny_penalty(clusters, adjacency_matrix, gamma=1.0, tau=1.0):
#     intra_sum = 0.0
#     sizes = [len(c) for c in clusters if len(c) > 0]
#     K = len(sizes)
#     n = sum(sizes)

#     # Intra-cluster compactness term
#     for cluster in clusters:
#         size = len(cluster)
#         if size == 0:
#             continue
#         dist = sum(adjacency_matrix[i, j] for i in cluster for j in cluster)
#         intra_sum += dist / (size * size)

#     # Unbalanced cluster penalty
#     s_bar = n / K
#     sigma_s = np.std(sizes, ddof=0)
#     s_thresh = s_bar - gamma * sigma_s

#     penalty = np.mean([
#         np.exp((max(0, s_thresh - s_c)) / tau)
#         for s_c in sizes
#     ])

  
#     return intra_sum + 0.01*penalty, intra_sum, penalty


# # Grid search over alpha
# alpha_grid = np.linspace(0.05, 1.00, 20)
# gamma_penalty = 1.0
# tau_penalty = 1.0
# cluster_counts = []
# cluster_losses = []

# for a in alpha_grid:
#     clusters = hierarchical_clustering(copy.deepcopy(adj_mat), thresh=a, linkage='average')
#     score, intra, penalty = clustering_loss_with_tiny_penalty(clusters, copy.deepcopy(adj_mat), gamma_penalty, tau_penalty)
#     cluster_counts.append(len(clusters))
#     cluster_losses.append(min(score,1))

# # Plot the figure with larger font sizes
# fig, ax1 = plt.subplots()

# # Right y-axis (number of clusters)
# ax2 = ax1.twinx()
# ax2.bar(alpha_grid, cluster_counts, width=0.03, color='blue', alpha=0.7)
# ax2.set_ylabel('Number of clusters', color='blue', fontsize=19)
# ax2.tick_params(axis='y', labelcolor='blue', labelsize=17)

# # Left y-axis (clustering loss)
# ax1.plot(alpha_grid, cluster_losses, color='red', marker='x', linestyle='--')
# ax1.set_ylabel(r'Clustering loss $\mathcal{L}_{\mathbb{C}}$', color='red', fontsize=19)
# ax1.tick_params(axis='y', labelcolor='red', labelsize=17)

# # X-axis (alpha)
# ax1.set_xlabel('Distance threshold (α)', fontsize=19)
# ax1.set_xticks(np.arange(0.0, 1.05, 0.1))
# ax1.tick_params(axis='x', labelsize=17)

# plt.grid(True)
# plt.tight_layout()

# # Save plot and data
# os.makedirs("plots", exist_ok=True)
# plt.savefig("plots/"+args.dataset+"_.png", dpi=300, bbox_inches='tight')
# np.savez(
#     f"plots/{args.dataset}_plot_data.npz",
#     alpha_grid=alpha_grid,
#     cluster_counts=np.array(cluster_counts),
#     cluster_losses=np.array(cluster_losses)
# )

# plt.show()


# In[7]:


#HR clustering 


import numpy as np
import copy
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


args.cluster_alpha=.4
#1.9
args.linkage = 'average' 
np.set_printoptions(precision=4)

cnt = args.num_users

print(f'Round {r}')
clients_idxs = np.arange(cnt)

adj_mat=A.detach().cpu().numpy()
#adj_mat=  normalize_matrix(v)*(1-grad_co) + grad_co*normalize_matrix((copy.deepcopy(Grad_similarites)))
#adj_mat=  normalize_matrix((copy.deepcopy(Grad_similarites)))
#adj_mat=  normalize_matrix(v)

clusters = hierarchical_clustering(copy.deepcopy(adj_mat), thresh=args.cluster_alpha, linkage='minimum')

cnt+= 10
print('')
print('Adjacency Matrix')
print(adj_mat)
print('')
print('Clusters: ')
print(clusters)
print('')
print(f'Number of Clusters {len(clusters)}')
print('')
for jj in range(len(clusters)):
    print(f'Cluster {jj}: {len(clusters[jj])} ')
    

clients_clust_id = {i:None for i in range(args.num_users)}
for i in range(args.num_users):
    for j in range(len(clusters)):
        if i in clusters[j]:
            clients_clust_id[i] = j
           # print(i, " Client in", "Cluster ", j, traindata_cls_counts[i] )
            break
print(f'Clients: Cluster_ID \n{clients_clust_id}')
for k in range(len(clusters)):
    print(f"Cluster {k}, First Client {clusters[k][0]}:", traindata_cls_counts[clusters[k][0]])



import numpy as np
from typing import Dict, List

def build_similarity_graph_rank_supply_nonzero(
        traindata_cls_counts: Dict[int, Dict[int, int]],
        clusters: List[List[int]],
        num_classes: int = 10,
        top_k: int = 2
) -> Dict[int, List[int]]:
    """
    Build a directed Cluster-Complementarity Graph.
    Demand is high for rare classes; supply is high for abundant classes.

    Parameters
    ----------
    traindata_cls_counts : {client_id: {class: count}}
    clusters             : list of clusters (each is a list of client IDs)
    num_classes          : total number of classes
    top_k                : keep this many outgoing edges per cluster

    Returns
    -------
    similarity_graph : {cluster_id: [target clusters]}
    """
    C, K = len(clusters), num_classes

    # ----------------------------------------------------------
    # 1. Per-client rank over *non-zero* classes
    # ----------------------------------------------------------
    client_rank = [{} for _ in range(len(traindata_cls_counts))]
    for cid, counts in traindata_cls_counts.items():
        present = [(cls, cnt) for cls, cnt in counts.items() if cnt > 0]
        if not present:
            continue                                # client has no data
        present.sort(key=lambda x: x[1])            # ascending (rarest first)
        for r, (cls, _) in enumerate(present):
            client_rank[cid][cls] = r               # rank 0 … m_i-1

    # ----------------------------------------------------------
    # 2. Demand weight  w[p,k]   (large if class k is rare in cluster p)
    # ----------------------------------------------------------
    w = np.zeros((C, K), dtype=float)
    for p, clist in enumerate(clusters):
        for cid in clist:
            m_i = len(client_rank[cid])             # number of classes present
            for cls, r in client_rank[cid].items():
                w[p, cls] += (m_i - r)              # larger when rarer

    # ----------------------------------------------------------
    # 3. Supply strength s[q,k] (large if class k is abundant in cluster q)
    # ----------------------------------------------------------
    s = np.zeros((C, K), dtype=float)
    for q, clist in enumerate(clusters):
        for cid in clist:
            for cls, r in client_rank[cid].items():
                s[q, cls] += (r + 1)                # larger when more common
        n_q = max(len(clist), 1)
        s[q] /= n_q                                 # per-client normalisation

    # ----------------------------------------------------------
    # 4. Complementarity score matrix  A = W · Sᵀ
    # ----------------------------------------------------------
    score = w @ s.T
    np.fill_diagonal(score, -np.inf)                # forbid self-loops

    # ----------------------------------------------------------
    # 5. Keep top-k outgoing edges for each cluster
    # ----------------------------------------------------------
    graph = {p: np.argsort(-score[p])[:top_k].tolist() for p in range(C)}
    return graph
similarity_graph = build_similarity_graph_rank_supply_nonzero(
    traindata_cls_counts,
    clusters,
    num_classes=10,
    top_k=2
)

# convert “out-list” format {p:[q1,q2]}  ➜  incoming list  {q:[p1,p2]}
H_out = {z: [] for z in range(len(clusters))}
for src, tgts in similarity_graph.items():
    for t in tgts:
        H_out[t].append(src)        # cluster src  ➜  learns from  ➜  t


# In[8]:


#Goes to utility
def clone_encoder(encoder_template: nn.Module) -> nn.Module:
    """
    Return a fresh deep copy of the encoder (with the same structure + random weights).
    """
    return copy.deepcopy(encoder_template).to(next(encoder_template.parameters()).device)







# ------------------------------------------------------------
# Robust FedAvg over a list of encoders (any architecture)
# ------------------------------------------------------------
def fedavg_delta(delta_list, weights):
    """
    FedAvg for *additive updates*  Δθ.
    Keys are already identical across dicts.
    """
    tot = float(sum(weights))
    out = copy.deepcopy(delta_list[0])
    with torch.no_grad():
        for k in out:
            out[k].zero_()
            for w, d in zip(weights, delta_list):
                out[k] += (w / tot) * d[k]
    return out







def fedavg_encoder(encoder_list: list[nn.Module]) -> nn.Module:
    """
    Return a copy of encoder_list[0] whose parameters are the
    element-wise average of all encoders in the list.
    """
    n = len(encoder_list)
    assert n > 0, "encoder_list must be non-empty"

    # clone first encoder as template
    avg_enc = copy.deepcopy(encoder_list[0])
    avg_sd  = avg_enc.state_dict()           # OrderedDict of tensors

    with torch.no_grad():
        for k in avg_sd.keys():
            stacked = torch.stack([enc.state_dict()[k] for enc in encoder_list], dim=0)
            avg_sd[k].copy_(stacked.mean(dim=0))

    avg_enc.load_state_dict(avg_sd)
    return avg_enc


def make_combined_model_from_single(
    single_model: nn.Module,
    cluster_enc: nn.Module,
    num_classes: int = 10,
    dataset: str = "mnist"
) -> nn.Module:
    """
    single_model : instance of SimpleCNNMNIST2 or SimpleCNN
    cluster_enc  : averaged encoder to use as PRIMARY encoder
    dataset      : string, e.g., 'mnist', 'fmnist', 'svhn', 'cifar10', 'cifar100'
    """
    # fresh secondary encoder (same dim) – random init
    #sec_enc = clone_encoder(single_model.encoder)
    sec_enc = copy.deepcopy(cluster_enc)  # same weights as primary

    # input dim to classifier
    in_dim = 84 if dataset in {"mnist", "fmnist"} else 256
    clf_input_dim = 2 * in_dim
    new_clf = nn.Linear(clf_input_dim, num_classes)
    clf_sd = new_clf.state_dict()

    if dataset in {"mnist", "fmnist"}:
        return CombinedModelMNIST(cluster_enc, sec_enc, clf_sd, num_classes)
    elif dataset in {"svhn", "cifar10", "cifar100"}:
        return CombinedSimpleCNN(cluster_enc, sec_enc, clf_sd, num_classes)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")



def switch_cluster_to_combined(
    clients, 
    cluster_ids, 
    w_glob_per_cluster, 
    cluster_id, 
    num_classes=10,
    dataset="mnist"    # <<< pass the dataset name here
):
    """
    Initialize combined model for a cluster using FedAvg on primary encoders.
    
    clients             : list[Client_ClusterFL]
    cluster_ids         : list[int] – client indices in the SAME cluster
    w_glob_per_cluster  : list of cluster-level global state_dicts
    cluster_id          : int – ID of the current cluster
    num_classes         : int – number of output classes
    dataset             : str – 'mnist', 'fmnist', 'svhn', 'cifar10', 'cifar100'
    """
    # 1. collect trained encoders from clients
    encoders = [copy.deepcopy(clients[i].net.encoder).to(clients[i].device)
                for i in cluster_ids]

    # 2. FedAvg on primary encoder
    cluster_enc = fedavg_encoder(encoders)

    # 3. use the first client’s model as a template
    single_template = clients[cluster_ids[0]].net

    # 4. build the combined model (primary + frozen secondary + classifier)
    combined = make_combined_model_from_single(
        single_model=single_template,
        cluster_enc=cluster_enc,
        num_classes=num_classes,
        dataset=dataset
    )

    # 5. get state_dict of combined model
    combined_sd = combined.state_dict()

    # 6. assign combined model to all clients in the cluster
    for i in cluster_ids:
        clients[i].net = copy.deepcopy(combined)
        clients[i].net.load_state_dict(combined_sd)

    # 7. store cluster-level global model
    w_glob_per_cluster[cluster_id] = copy.deepcopy(combined_sd)


w_glob_per_cluster = [None] * len(clusters)


w_glob_per_cluster = [None] * len(clusters)
for z, clist in enumerate(clusters):
    switch_cluster_to_combined(
        clients=clients,
        cluster_ids=clist,
        w_glob_per_cluster=w_glob_per_cluster,
        cluster_id=z,
        num_classes=10,
        dataset=args.dataset   # <<< set this based on your current run
    )
print(">>> All clusters switched to dual-encoder ")



###################################### Clustered FL training 
# ============================================================
# Helpers
# ============================================================

from collections import OrderedDict

def cluster_participants(idxs_users, clients_clust_id):
    """
    Map sampled user IDs to {cluster_id: [user_ids]}.
    """
    out = {}
    for uid in idxs_users:
        cid = clients_clust_id[uid]
        out.setdefault(cid, []).append(uid)
    return out


def make_weight_vec(id_list, net_dataidx_map):
    """
    Return FedAvg weights proportional to each client's data size.
    """
    sizes = [len(net_dataidx_map[u]) for u in id_list]
    tot   = sum(sizes)
    return [s / tot for s in sizes]


def FedAvg_trainable(sd_list, weights):
    """
    FedAvg only the trainable blocks: own_encoder.* and classifier.*.
    Frozen secondary_* keys are left untouched.
    """
    out = copy.deepcopy(sd_list[0])
    with torch.no_grad():
        for k in out:
            if k.startswith(("own_encoder", "classifier")):
                out[k].zero_()
                for w, sd in zip(weights, sd_list):
                    out[k] += w * sd[k]
    return out

def fedavg_secondary(sd_list, weights):
    tot = float(sum(weights))
    out = copy.deepcopy(sd_list[0])
    with torch.no_grad():
        for k in out:                       # keys already 'secondary_encoder.*'
            out[k].zero_()
            for w, sd in zip(weights, sd_list):
                out[k] += (w / tot) * sd[k]
    return out

# ============================================================
# Metric bookkeeping
# ============================================================

client_metric = {
    uid: {"best_pre": 0.0, "best_post": 0.0, "best_any": 0.0}
    for uid in range(args.num_users)
}


# In[ ]:


# ============================================================
# Clustered-FL primary-update loop
# ============================================================
n_rounds   = 161
print_step = 10              # stats cadence after warm-up
loss_buf   = []

for rnd in range(n_rounds):

    print(f"\n##### ROUND {rnd+1} #####")

    # 1) sample users
    m            = int(args.frac * args.num_users)
    idxs_users   = np.random.choice(args.num_users, m, replace=False)
    clusters_now = cluster_participants(idxs_users, clients_clust_id)

    # 2) broadcast combined cluster model to each selected user
    for uid in idxs_users:
        cid = clients_clust_id[uid]
        clients[uid].set_state_dict(copy.deepcopy(w_glob_per_cluster[cid]))

    # 3) per-cluster FedAvg containers
    enc_pool, clf_pool, freq_pool = {}, {}, {}

    # 4) local training
    for uid in idxs_users:
        cid   = clients_clust_id[uid]
        ndata = len(net_dataidx_map[uid])

        # ---- metrics before train
        pre_loss, pre_acc = clients[uid].eval_test()
        client_metric[uid]["best_pre"]  = max(client_metric[uid]["best_pre"],  pre_acc)

        # ---- local SGD (own_encoder + classifier only)
        loss = clients[uid].train2()
        loss_buf.append(loss)

        # ---- metrics after train
        post_loss, post_acc = clients[uid].eval_test()
        client_metric[uid]["best_post"] = max(client_metric[uid]["best_post"], post_acc)
        client_metric[uid]["best_any"]  = max(client_metric[uid]["best_any"],
                                              pre_acc, post_acc)

        # ---- stash trainable weights for FedAvg
        # Wrap encoder weights with prefix
        own_sd = clients[uid].net.own_encoder.state_dict()
        own_sd_prefixed = OrderedDict({f"own_encoder.{k}": v for k, v in own_sd.items()})
        enc_pool.setdefault(cid, []).append(own_sd_prefixed)
        
        # Wrap classifier weights with prefix
        clf_sd = clients[uid].net.classifier.state_dict()
        clf_sd_prefixed = OrderedDict({f"classifier.{k}": v for k, v in clf_sd.items()})
        clf_pool.setdefault(cid, []).append(clf_sd_prefixed)

    # 5) FedAvg per cluster (trainable blocks)
    for cid in enc_pool:
        weights = make_weight_vec(clusters_now[cid], net_dataidx_map)
        upd_enc = FedAvg_trainable(enc_pool[cid], weights)
        upd_clf = FedAvg_trainable(clf_pool[cid], weights)
        w_glob_per_cluster[cid].update(upd_enc)
        w_glob_per_cluster[cid].update(upd_clf)
    
    # ------------------------------------------------------------------
    # 6)  Secondary-encoder enrichment  (ONLY for sampled clusters)

    local_sec_ep = 10
    sec_lr       = 0.01
    
    for z, clist_sampled in clusters_now.items():
    
        learners = H_out.get(z, [])
        if not learners or len(clist_sampled) == 0:
            continue
    
        # 6.1  Θ2′z  (average current learner weights)
        learner_secs = [
            {k: v.clone() for k, v in w_glob_per_cluster[j].items()
             if k.startswith('secondary_encoder.')}
            for j in learners
        ]
        θ2_prime = fedavg_secondary(learner_secs, [1] * len(learner_secs))
    
        # strip prefix for bare encoder
        base_sec_state = OrderedDict(
            (k.replace("secondary_encoder.", ""), v) for k, v in θ2_prime.items()
        )
    
        # 6.3  each sampled client returns Δθ_i (CPU tensors)
        delta_list, sizes = [], []
        for uid in clist_sampled:
            Δθ = clients[uid].train_secondary(base_sec_state,
                                              epochs=local_sec_ep,
                                              lr=sec_lr)
            delta_list.append(Δθ)
            sizes.append(len(net_dataidx_map[uid]))
    
        # 6.4  FedAvg over Δθ_i   →   Δ̄θ(z)  (still CPU tensors)
        Δ̄θ = fedavg_delta(delta_list, sizes)
    
        # 6.5  apply Δ̄θ(z) to every learner cluster j
        for j in learners:
            for k, dv in Δ̄θ.items():      # k = "0.weight", …
                key = f"secondary_encoder.{k}"   # full key in global state dict

                # ensure dv is on the same device as the target parameter
                target_param = w_glob_per_cluster[j][key]
                dv = dv.to(target_param.device)

                # you can do in-place add OR reassign; both are fine
                w_glob_per_cluster[j][key] = target_param + dv

    # 7) periodic statistics
    if rnd < 4:
        print_step = 1
    else:
        print_step = 10

    if rnd % print_step == 0:
        avg_loss = np.mean(loss_buf) if loss_buf else 0.0
        avg_pre  = np.mean([m["best_pre"]  for m in client_metric.values()])
        avg_post = np.mean([m["best_post"] for m in client_metric.values()])
        avg_any  = np.mean([m["best_any"]  for m in client_metric.values()])

        print("ROUND STATS")
        print(f"avg train loss : {avg_loss:.4f}")
        print(f"avg best PRE   : {avg_pre :.2f}%")
        print(f"avg best POST  : {avg_post:.2f}%")
        print(f"avg best ANY   : {avg_any :.2f}%\n")

        # optional per-client line (comment if too verbose)
        for uid in range(args.num_users):
            bm = client_metric[uid]
            print(f"Client {uid:3d} | best_pre={bm['best_pre'] :6.2f} "
                  f"| best_post={bm['best_post']:6.2f} "
                  f"| best_any={bm['best_any'] :6.2f}")

        # reset buffers for next report window
        loss_buf.clear()
        gc.collect()

