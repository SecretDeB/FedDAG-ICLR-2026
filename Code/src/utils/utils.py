import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import random
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from .datasets import MNIST_truncated, MNIST_rotated, CIFAR10_truncated, CIFAR10_rotated, CIFAR100_truncated, SVHN_custom, FashionMNIST_truncated, CustomTensorDataset, CelebA_custom, FEMNIST, Generated, genData, ImageFolder_custom, USPS_truncated

from math import sqrt

import torch.nn as nn

import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import Subset
import time
import random
import copy 
import matplotlib.pyplot as plt

import sklearn.datasets as sk
from sklearn.datasets import load_svmlight_file
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import math

from collections import defaultdict

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def load_usps_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    usps_train_ds = USPS_truncated(datadir, train=True, download=True, transform=transform)
    usps_test_ds = USPS_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = usps_train_ds.data, usps_train_ds.target
    X_test, y_test = usps_test_ds.data, usps_test_ds.target

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return (X_train, y_train, X_test, y_test)

def load_mnist_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_mnist_rotated_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_rotated(datadir, rotation=0, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_rotated(datadir, rotation=0, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_fmnist_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = FashionMNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = FashionMNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_svhn_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    svhn_train_ds = SVHN_custom(datadir, train=True, download=True, transform=transform)
    svhn_test_ds = SVHN_custom(datadir, train=False, download=True, transform=transform)

    X_train, y_train = svhn_train_ds.data, svhn_train_ds.target
    X_test, y_test = svhn_test_ds.data, svhn_test_ds.target

    # X_train = X_train.data.numpy()
    # y_train = y_train.data.numpy()
    # X_test = X_test.data.numpy()
    # y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_cifar10_rotated_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_rotated(datadir, rotation=0, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_rotated(datadir, rotation=0, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_cifar100_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_tinyimagenet_data(datadir):
    print(datadir)
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_custom(datadir+'tiny-imagenet-200/train/', transform=transform)
    xray_test_ds = ImageFolder_custom(datadir+'tiny-imagenet-200/val/', transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)

def load_celeba_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    celeba_train_ds = CelebA_custom(datadir, split='train', target_type="attr", download=True, transform=transform)
    celeba_test_ds = CelebA_custom(datadir, split='test', target_type="attr", download=True, transform=transform)

    gender_index = celeba_train_ds.attr_names.index('Male')
    y_train =  celeba_train_ds.attr[:,gender_index:gender_index+1].reshape(-1)
    y_test = celeba_test_ds.attr[:,gender_index:gender_index+1].reshape(-1)

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (None, y_train, None, y_test)

def load_femnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = FEMNIST(datadir, train=True, transform=transform, download=True)
    mnist_test_ds = FEMNIST(datadir, train=False, transform=transform, download=True)

    X_train, y_train, u_train = mnist_train_ds.data, mnist_train_ds.targets, mnist_train_ds.users_index
    X_test, y_test, u_test = mnist_test_ds.data, mnist_test_ds.targets, mnist_test_ds.users_index

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    u_train = np.array(u_train)
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()
    u_test = np.array(u_test)

    return (X_train, y_train, u_train, X_test, y_test, u_test)

def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    #logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts

def generate_sequence_from_unselected(iteration, num, unselected_labels, K):
    sequence = []

    # If num is smaller than unselected labels, randomly select num from unselected labels
    if len(unselected_labels) >= num:
        sequence = np.random.choice(unselected_labels, size=num, replace=False).tolist()
    else:
        # Select all unselected labels first
        sequence = np.random.choice(unselected_labels, size=len(unselected_labels), replace=False).tolist()

        # Fill up the remaining slots with unique random labels from the full set K, excluding already selected ones
        remaining_needed = num - len(sequence)
        all_labels = set(range(K))  # Create a set of all possible labels
        remaining_choices = list(all_labels - set(sequence))  # Exclude already selected labels
        
        # Randomly select the remaining unique labels
        remaining_sequence = np.random.choice(remaining_choices, size=remaining_needed, replace=False).tolist()
        sequence.extend(remaining_sequence)

    return sequence

    
def generate_sequence(first_class, num, total_classes):
    available_classes = list(range(total_classes))
    available_classes.remove(first_class)
    sequence = [first_class] + random.sample(available_classes, num-1)
    return sequence

import numpy as np

def split_array_dirichlet(idx_k, times, alpha):
    proportions = np.random.dirichlet(alpha=np.ones(times) * alpha)

    sizes = np.round(proportions * len(idx_k)).astype(int)
    sizes = np.maximum(sizes, 1)
    while np.sum(sizes) != len(idx_k):
        if np.sum(sizes) > len(idx_k):
            sizes[np.argmax(sizes)] -= np.sum(sizes) - len(idx_k)
        else:
            sizes[np.argmin(sizes)] += len(idx_k) - np.sum(sizes)

    indices = np.cumsum(sizes)[:-1]
    
    split = np.split(idx_k, indices)

    return split



def partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.4, local_view=False):
    #np.random.seed(2020)
    #torch.manual_seed(2020)

    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    elif data == 'mnist_rotated':
        X_train, y_train, X_test, y_test = load_mnist_rotated_data(datadir)
    elif dataset == 'usps':
        X_train, y_train, X_test, y_test = load_usps_data(datadir)
    elif dataset == 'fmnist':
        X_train, y_train, X_test, y_test = load_fmnist_data(datadir)
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'cifar10_rotated':
        X_train, y_train, X_test, y_test = load_cifar10_rotated_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'tinyimagenet':
        X_train, y_train, X_test, y_test = load_tinyimagenet_data(datadir)
    elif dataset == 'svhn':
        X_train, y_train, X_test, y_test = load_svhn_data(datadir)
    elif dataset == 'celeba':
        X_train, y_train, X_test, y_test = load_celeba_data(datadir)
    elif dataset == 'femnist':
        X_train, y_train, u_train, X_test, y_test, u_test = load_femnist_data(datadir)
    elif dataset == 'generated':
        X_train, y_train = [], []
        for loc in range(4):
            for i in range(1000):
                p1 = random.random()
                p2 = random.random()
                p3 = random.random()
                if loc > 1:
                    p2 = -p2
                if loc % 2 ==1:
                    p3 = -p3
                if i % 2 == 0:
                    X_train.append([p1, p2, p3])
                    y_train.append(0)
                else:
                    X_train.append([-p1, -p2, -p3])
                    y_train.append(1)
        X_test, y_test = [], []
        for i in range(1000):
            p1 = random.random() * 2 - 1
            p2 = random.random() * 2 - 1
            p3 = random.random() * 2 - 1
            X_test.append([p1, p2, p3])
            if p1>0:
                y_test.append(0)
            else:
                y_test.append(1)
        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)
        y_test = np.array(y_test, dtype=np.int64)
        idxs = np.linspace(0,3999,4000,dtype=np.int64)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
        mkdirs("data/generated/")
        np.save("data/generated/X_train.npy",X_train)
        np.save("data/generated/X_test.npy",X_test)
        np.save("data/generated/y_train.npy",y_train)
        np.save("data/generated/y_test.npy",y_test)
    
    #elif dataset == 'covtype':
    #    cov_type = sk.fetch_covtype('./data')
    #    num_train = int(581012 * 0.75)
    #    idxs = np.random.permutation(581012)
    #    X_train = np.array(cov_type['data'][idxs[:num_train]], dtype=np.float32)
    #    y_train = np.array(cov_type['target'][idxs[:num_train]], dtype=np.int32) - 1
    #    X_test = np.array(cov_type['data'][idxs[num_train:]], dtype=np.float32)
    #    y_test = np.array(cov_type['target'][idxs[num_train:]], dtype=np.int32) - 1
    #    mkdirs("data/generated/")
    #    np.save("data/generated/X_train.npy",X_train)
    #    np.save("data/generated/X_test.npy",X_test)
    #    np.save("data/generated/y_train.npy",y_train)
    #    np.save("data/generated/y_test.npy",y_test)

    elif dataset in ('rcv1', 'SUSY', 'covtype'):
        X_train, y_train = load_svmlight_file("../../../data/{}".format(dataset))
        X_train = X_train.todense()
        num_train = int(X_train.shape[0] * 0.75)
        if dataset == 'covtype':
            y_train = y_train-1
        else:
            y_train = (y_train+1)/2
        idxs = np.random.permutation(X_train.shape[0])

        X_test = np.array(X_train[idxs[num_train:]], dtype=np.float32)
        y_test = np.array(y_train[idxs[num_train:]], dtype=np.int32)
        X_train = np.array(X_train[idxs[:num_train]], dtype=np.float32)
        y_train = np.array(y_train[idxs[:num_train]], dtype=np.int32)

        mkdirs("data/generated/")
        np.save("data/generated/X_train.npy",X_train)
        np.save("data/generated/X_test.npy",X_test)
        np.save("data/generated/y_train.npy",y_train)
        np.save("data/generated/y_test.npy",y_test)

    elif dataset in ('a9a'):
        X_train, y_train = load_svmlight_file("../../../data/{}".format(dataset))
        X_test, y_test = load_svmlight_file("../../../data/{}.t".format(dataset))
        X_train = X_train.todense()
        X_test = X_test.todense()
        X_test = np.c_[X_test, np.zeros((len(y_test), X_train.shape[1] - np.size(X_test[0, :])))]

        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = (y_train+1)/2
        y_test = (y_test+1)/2
        y_train = np.array(y_train, dtype=np.int32)
        y_test = np.array(y_test, dtype=np.int32)

        mkdirs("data/generated/")
        np.save("data/generated/X_train.npy",X_train)
        np.save("data/generated/X_test.npy",X_test)
        np.save("data/generated/y_train.npy",y_train)
        np.save("data/generated/y_test.npy",y_test)

    n_train = y_train.shape[0]

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}  
        
    elif partition == "lda":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            # min_require_size = 100
        elif dataset in ('cifar100'):
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200

        N = y_train.shape[0]
        #np.random.seed(2021)
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                # logger.info("proportions1: ", proportions)
                # logger.info("sum pro1:", np.sum(proportions))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                # logger.info("proportions2: ", proportions)
                proportions = proportions / proportions.sum()
                # logger.info("proportions3: ", proportions)
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # logger.info("proportions4: ", proportions)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break


        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

   
    elif partition == "flag-non-iid":
        K = 10
        x=5 #underlying different group of clients
        num = 3 #class skew, number of labels
        if dataset == 'cifar100':
            K = 100
            x=3
            num=30
        iterations = 10000
        times = [0 for i in range(K)] 
        net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        contain = {i: [] for i in range(n_parties)} 
    
        selected_labels = set()  # Track which labels (classes K) have been selected
        cnt = 0
    
        for i in range(x):
            # If not all labels have been selected, prioritize the unselected ones
            if len(selected_labels) < K:
                # Identify unselected labels
                unselected_labels = list(set(range(K)) - selected_labels)
                # Generate sequence prioritizing unselected labels
                sequence = generate_sequence_from_unselected(i, num, unselected_labels, K)
                # Add the newly selected labels to the selected set
                selected_labels.update(sequence)
            else:
                # If all labels have been selected, generate the sequence randomly
                sequence = generate_sequence(i, num, K)

            
          
            flag = 0
            for client_id in range(n_parties):
                if len(contain[client_id]) == 0:
                    if random.random() < 1 / (x - i): 
                        flag = 1
                        contain[client_id].extend(sequence)
                        for cls in sequence:
                            times[cls] += 1
            cnt += flag
    
        for i in range(K):
            if times[i] > 0:
                idx_k = np.where(y_train == i)[0]
                np.random.shuffle(idx_k)
                split = split_array_dirichlet(idx_k, times[i], 1)  # split evenly
                
                ids = 0
                for j in range(n_parties):
                    if i in contain[j]:
                        net_dataidx_map[j] = np.append(net_dataidx_map[j], split[ids])
                        ids += 1

    
    
    
    print(f'partition: {partition}')
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    print('Data statistics Train:\n %s \n' % str(traindata_cls_counts))
    
    if local_view:
        net_dataidx_map_test = {i: [] for i in range(n_parties)}
        for k_id, stat in traindata_cls_counts.items():
            labels = list(stat.keys())
            for l in labels:
                idx_k = np.where(y_test==l)[0]
                net_dataidx_map_test[k_id].extend(idx_k.tolist())

        testdata_cls_counts = record_net_data_stats(y_test, net_dataidx_map_test, logdir)
        print('Data statistics Test:\n %s \n' % str(testdata_cls_counts))
    else: 
        net_dataidx_map_test = None 
        testdata_cls_counts = None 

    return (X_train, y_train, X_test, y_test, net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts)

def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu"):

    was_training = False
    if model.training:
        model.eval()
        was_training = True
        
    model.to(device)
    
    w = model.state_dict()
    name = list(w.keys())[0]
    print(f'COMP ACC {w[name][0,0,0]}')
            
    true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total = 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                out = model(x)
                _, pred_label = torch.max(out.data, 1)

                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()
                
                #pred = out.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                #correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct/float(total), conf_matrix

    return correct/float(total)

def save_model(model, model_index, args):
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir+"trained_local_model"+str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return

def load_model(model, model_index, device="cpu"):
    with open("trained_local_model"+str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., net_id=None, total=0):
        self.std = std
        self.mean = mean
        self.net_id = net_id
        self.num = int(sqrt(total))
        if self.num * self.num < total:
            self.num = self.num + 1

    def __call__(self, tensor):
        if self.net_id is None:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            tmp = torch.randn(tensor.size())
            filt = torch.zeros(tensor.size())
            size = int(28 / self.num)
            row = int(self.net_id / size)
            col = self.net_id % size
            for i in range(size):
                for j in range(size):
                    filt[:,row*size+i,col*size+j] = 1
            tmp = tmp * filt
            return tensor + tmp * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0,
                   net_id=None, total=0, dataidxs_test=None,
                   same_size=False, target_transform=None, rotation=0,
                   num_workers: int = 4,
                   pin_memory: bool = False):
    flag = 1 if dataidxs is not None else 0
    if dataset in ('mnist', 'mnist_rotated', 'femnist', 'fmnist', 'cifar10', 'cifar10_rotated', 'cifar100',
                   'svhn', 'tinyimagenet', 'generated', 'covtype', 'a9a', 'rcv1', 'SUSY', 'usps'):
        if dataset == 'mnist' or dataset == 'mnist_rotated':
            if dataset == 'mnist':
                dl_obj = MNIST_truncated
            elif dataset == 'mnist_rotated':
                dl_obj = MNIST_rotated
            
            if same_size:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Pad(2, fill=0, padding_mode='constant'),
                    transforms.Lambda(lambda x: x.repeat(3,1,1)),
                    AddGaussianNoise(0., noise_level, net_id, total), 
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Pad(2, fill=0, padding_mode='constant'),
                    transforms.Lambda(lambda x: x.repeat(3,1,1)),
                    AddGaussianNoise(0., noise_level, net_id, total),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            else: 
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    AddGaussianNoise(0., noise_level, net_id, total), 
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    AddGaussianNoise(0., noise_level, net_id, total),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

        elif dataset == 'femnist':
            dl_obj = FEMNIST
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total),
                transforms.Normalize((0.1307,), (0.3081,))
             ])
            
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total), 
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        elif dataset == 'fmnist':
            dl_obj = FashionMNIST_truncated
            
            if same_size:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Pad(2, fill=0, padding_mode='constant'),
                    transforms.Lambda(lambda x: x.repeat(3,1,1)),
                    AddGaussianNoise(0., noise_level, net_id, total), 
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Pad(2, fill=0, padding_mode='constant'),
                    transforms.Lambda(lambda x: x.repeat(3,1,1)),
                    AddGaussianNoise(0., noise_level, net_id, total), 
                    transforms.Normalize((0.1307,), (0.3081,))
                 ])
            else: 
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    AddGaussianNoise(0., noise_level, net_id, total), 
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    AddGaussianNoise(0., noise_level, net_id, total), 
                    transforms.Normalize((0.1307,), (0.3081,))
                 ])

        elif dataset == 'svhn':
            dl_obj = SVHN_custom
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
            ])

        elif dataset == 'cifar10' or dataset == 'cifar10_rotated':
            if dataset == 'cifar10': 
                dl_obj = CIFAR10_truncated
            elif dataset == 'cifar10_rotated':
                dl_obj = CIFAR10_rotated

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Lambda(lambda x: F.pad(
                #    Variable(x.unsqueeze(0), requires_grad=False),
                #    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                #transforms.ToPILImage(),
                #transforms.RandomCrop(32),
                #transforms.RandomHorizontalFlip(),
                #transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total), 
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total), 
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        elif dataset == 'cifar100':
            dl_obj = CIFAR100_truncated

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Lambda(lambda x: F.pad(
                #    Variable(x.unsqueeze(0), requires_grad=False),
                #    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                #transforms.ToPILImage(),
                #transforms.RandomCrop(32),
                #transforms.RandomHorizontalFlip(),
                #transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total), 
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total), 
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
            
        elif dataset == 'tinyimagenet':
            dl_obj = ImageFolder_custom
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            train_ds = dl_obj(datadir+'tiny-imagenet-200/train/', dataidxs=dataidxs, transform=transform_train)
            test_ds = dl_obj(datadir+'tiny-imagenet-200/val/', transform=transform_test)

            train_dl = data.DataLoader(
                dataset=train_ds,
                batch_size=train_bs,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            
            test_dl = data.DataLoader(
                dataset=test_ds,
                batch_size=test_bs,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

            
        elif dataset == 'usps': 
            dl_obj = USPS_truncated
            
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Pad(8, fill=0, padding_mode='constant'),
                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                AddGaussianNoise(0., noise_level, net_id, total), 
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Pad(8, fill=0, padding_mode='constant'),
                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                AddGaussianNoise(0., noise_level, net_id, total), 
                transforms.Normalize((0.1307,), (0.3081,))
             ])

        else:
            dl_obj = Generated
            transform_train = None
            transform_test = None
        
        if dataset != 'tinyimagenet':
            if dataset == 'mnist_rotated' or dataset == 'cifar10_rotated':
                train_ds = dl_obj(datadir, rotation=rotation, dataidxs=dataidxs, train=True, transform=transform_train, 
                              target_transform=target_transform, download=True)
                test_ds = dl_obj(datadir, rotation=rotation, dataidxs=dataidxs_test, train=False, transform=transform_test, 
                             target_transform=target_transform, download=True)
            else:
                train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, 
                                  target_transform=target_transform, download=True)
                test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, 
                                 target_transform=target_transform, download=True)
                #print(len(train_ds))
            
                if flag == 1:
                    # Make a deep copy of train_ds
                    train_ds2 = copy.deepcopy(train_ds)
                    
                    # Dictionary to hold indices by class
                    class_indices = defaultdict(list)
                    
                    # Populate the dictionary with indices sorted by target
                    for idx in range(len(train_ds2)):
                        target = train_ds2[idx][1]  # Assuming train_ds2[idx] returns (image, target)
                        if torch.is_tensor(target):
                            target = int(target.numpy())
                        class_indices[target].append(idx)
                    
                    validation_list = []
                    pre_training_list = []
                    #print(class_indices)
                    # Shuffle and split each class separately
                    for class_idx, indices in class_indices.items():
                        random.shuffle(indices)
                        split_idx = math.ceil(len(indices) * 0.05)  # Using ceil to round up
                        
                        validation_list.extend(indices[:split_idx])
                        pre_training_list.extend(indices[split_idx:])
                    
                    # Create subsets for pre-training and validation
                    #print("pre-trainlist",pre_training_list) 
                    pre_train_ds = Subset(train_ds2, pre_training_list)
                    validation_ds = Subset(train_ds2, validation_list)
                
                    # Create DataLoaders for pre-training and validation datasets
                    #print("hell")
                    #print(len(pre_train_ds))
                    pre_train_dl = data.DataLoader(
                        dataset=pre_train_ds,
                        batch_size=train_bs,
                        shuffle=True,
                        drop_last=False,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                    )
                    
                    validation_dl = data.DataLoader(
                        dataset=validation_ds,
                        batch_size=test_bs,
                        shuffle=False,
                        drop_last=False,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                    )

            train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
            test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)
            # if(flag==1):
            #     print(len(pre_train_ds))
            #     pre_train_dl = data.DataLoader(dataset=pre_train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
            #     validation_dl = data.DataLoader(dataset=validation_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    if(flag==0):
        return train_dl, test_dl, train_ds, test_ds
        
    return train_dl, test_dl, train_ds, test_ds, pre_train_dl, validation_dl


def weights_init(m):
    """
    Initialise weights of the model.
    """
    if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif(type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.

    Treating Q(cj | x) as a factored Gaussian.
    """
    def __call__(self, x, mu, var):

        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll


def noise_sample(choice, n_dis_c, dis_c_dim, n_con_c, n_z, batch_size, device):
    """
    Sample random noise vector for training.

    INPUT
    --------
    n_dis_c : Number of discrete latent code.
    dis_c_dim : Dimension of discrete latent code.
    n_con_c : Number of continuous latent code.
    n_z : Dimension of iicompressible noise.
    batch_size : Batch Size
    device : GPU/CPU
    """

    z = torch.randn(batch_size, n_z, 1, 1, device=device)
    idx = np.zeros((n_dis_c, batch_size))
    if(n_dis_c != 0):
        dis_c = torch.zeros(batch_size, n_dis_c, dis_c_dim, device=device)

        c_tmp = np.array(choice)

        for i in range(n_dis_c):
            idx[i] = np.random.randint(len(choice), size=batch_size)
            for j in range(batch_size):
                idx[i][j] = c_tmp[int(idx[i][j])]

            dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0

        dis_c = dis_c.view(batch_size, -1, 1, 1)

    if(n_con_c != 0):
        # Random uniform between -1 and 1.
        con_c = torch.rand(batch_size, n_con_c, 1, 1, device=device) * 2 - 1

    noise = z
    if(n_dis_c != 0):
        noise = torch.cat((z, dis_c), dim=1)
    if(n_con_c != 0):
        noise = torch.cat((noise, con_c), dim=1)

    return noise, idx


#for gradient similarity 

def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])

  
# def pairwise_angles(sources):
#     angles = np.zeros([len(sources), len(sources)])
#     for i, source1 in enumerate(sources):
#         for j, source2 in enumerate(sources):
#             s11 = flatten(source1)
#             s22 = flatten(source2)
#             s1= s11.detach().numpy()
#             s2= s22.detach().numpy()
#             dot_product = np.sum(s1 * s2)
#             norm_product = np.linalg.norm(s1) * np.linalg.norm(s2)
#             # Clip dot product values between -1 and 1
#             dot_product = np.clip(dot_product / (norm_product + 1e-12), -1.0, 1.0)
#             # Compute angle in radians using arccosine
#             angle_radians = np.arccos(dot_product)
#             # Store angle in degrees in the angles tensor
#             angles[i, j] = angle_radians*180/np.pi
#     # Print the first 10x10 square of the array
#     print("Grad_similarites :")
#     print(angles)

#     return angles

# ---added new pairwise function raplacing the 940-960 lines: gradient compression objective
#   Assosscation: compress_gradient_with_mask, generate_sparsity_mask, Modified pairwise_angles
#   Assosscation: Modified: flatten + sparsify in gradient similarity
#   This is created to handle new compressed gradient compared ot previous
#   pairwise_angles stays the same (it now takes 1D numpy arrays) ---

def pairwise_angles(sources: list[np.ndarray]) -> np.ndarray:
    angles = np.zeros((len(sources), len(sources)))
    for i, a in enumerate(sources):
        for j, b in enumerate(sources):
            if i == j:
                continue  # leave diagonal as zero
            dot = np.dot(a, b)
            norm = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
            cos = np.clip(dot / norm, -1.0, 1.0)
            angles[i, j] = np.degrees(np.arccos(cos))
    print("Grad_similarities:")
    np.set_printoptions(suppress=True, precision=4, floatmode="fixed")
    print(angles)
    return angles

# # --- Added: generate a fixed mask once per run: gradient compression objective
def generate_sparsity_mask(dim: int, compression_ratio: int) -> np.ndarray:
    k = dim // compression_ratio
    return np.random.choice(dim, k, replace=False)

# # --- Added: apply the mask to a flattened gradient: gradient compression objective
def compress_gradient_with_mask(flat_grad: np.ndarray, mask: np.ndarray) -> np.ndarray:
   return flat_grad[mask]


def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone()-subtrahend[name].data.clone()
#end of gradient similarity 

def show_tensor_image(image_tensor):
    
    # Convert the tensor to numpy array
    np_image = image_tensor.numpy()

    # If your tensor image is normalized around zero, re-normalize it to [0, 1]
    #np_image = (np_image + 1) / 2

    # Ensure the values are within [0, 1]
    #np_image = np.clip(np_image, 0, 1)

    # Transpose the image dimensions from (C, H, W) to (H, W, C)
    np_image = np.transpose(np_image, (1, 2, 0))

    # Plot the image
    plt.imshow(np_image)
    plt.show()


def show_numpy_image(np_image):
    
    # Convert the tensor to numpy array
    #np_image = image_tensor.numpy()

    # If your tensor image is normalized around zero, re-normalize it to [0, 1]
    #np_image = (np_image + 1) / 2

    # Ensure the values are within [0, 1]
    #np_image = np.clip(np_image, 0, 1)

    # Transpose the image dimensions from (C, H, W) to (H, W, C)
    #np_image = np.transpose(np_image, (1, 2, 0))

    # Plot the image
    plt.imshow(np_image)
    plt.show()
def normalize_images(images):
    """
    Normalize the images by centering them.
    
    :param images: A numpy array of shape (N, 32, 32, 3) representing N images
    :return: A numpy array of the normalized images
    """
    # Calculate the mean and std deviation across all images for each channel
    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1, 2))
    
    # Normalize the images
    normalized_images = (images - mean) / std
    
    return normalized_images
    
def min_max_normalize_image(image):
    """
    Normalize the image data to be in the range [0, 1] using min-max normalization.

    :param image: A numpy array representing an image.
    :return: Normalized image
    """
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image


#similarity calcuation matrices
def freq_similarity(traindata_cls_counts):
    """
    Calculate the Euclidean distance between all pairs of clients based on their class frequencies.
    
    Args:
    - traindata_cls_counts (dict): Dictionary containing class frequencies for each client.
    
    Returns:
    - dist_matrix (numpy.ndarray): Matrix containing Euclidean distances between all pairs of clients.
    """
    # Initialize an empty matrix to store distances
    num_clients = len(traindata_cls_counts)
    dist_matrix = np.zeros((num_clients, num_clients))

    # Calculate Euclidean distance between each pair of clients
    for i in range(num_clients):
        for j in range(num_clients):
            freq1 = np.zeros(10)
            freq2 = np.zeros(10)

            # Get class frequencies for client i
            for key, value in traindata_cls_counts[i].items():
                freq1[key] = value

            # Get class frequencies for client j
            for key, value in traindata_cls_counts[j].items():
                freq2[key] = value

            # Calculate Euclidean distance
            dist_matrix[i][j] = np.linalg.norm(freq1 - freq2)

    return dist_matrix


def select_representative_images(images,row_num, k, n_dimensions, label):
    #gets imgages of same label, 
    #returns representive images in tuples of image, label
    # Dimensionality reduction with SVD
    svd = TruncatedSVD(n_components=n_dimensions) 
    images1 = images.reshape(row_num, -1) #flattens images
    reduced_features = svd.fit_transform(images1) #reduce dimention

    # Clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(reduced_features)

    # Select representative images from each cluster
    representative_images = []
    for cluster_id in range(k):
        cluster_indices = np.where(clusters == cluster_id)[0]   #images indices of cluster i
        centroid_index = np.argmin(np.linalg.norm(reduced_features[cluster_indices] - np.mean(reduced_features[cluster_indices], axis=0), axis=1))
        representative_images.append((images[cluster_indices[centroid_index]],label))


    return representative_images

def select_representative_images_with_svd(images, k, label):
    # Flatten the images to create a 2D array
    flattened_images = np.array([image.flatten() for image in images])

    # Compute SVD
    U, s, Vt = np.linalg.svd(flattened_images, full_matrices=False)

    # Select representative images from the reduced space
    #representative_indices = np.argsort(s)[:k]
    representative_indices = np.argsort(s)[::-1][:k]
    representative_images = [(images[i],label) for i in representative_indices]

    
    return representative_images


#normalize gradient similairty and data similrity matrix

def normalize_matrix(matrix):
    # Create a mask for the main diagonal
    mask = np.eye(matrix.shape[0], dtype=bool)
    
    # Extract the non-diagonal elements
    non_diagonal_elements = matrix[~mask]
    
    # Find the minimum and maximum values among the non-diagonal elements
    min_val = np.min(non_diagonal_elements)
    max_val = np.max(non_diagonal_elements)
  
    
    # Normalize the matrix ignoring the diagonal
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    
    # Restore the original diagonal values
    normalized_matrix[mask] = matrix[mask]
    
    return normalized_matrix




