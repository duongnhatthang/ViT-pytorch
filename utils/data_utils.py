import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler


logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None
    elif args.dataset == "mri":
    #     augmentor = Compose([
    #     transforms.Lambda(lambda x: torch.Tensor(x)),
    #     RandomRotation(25),
    #     # RandomTranslate([0.11, 0.11]),
    #     RandomHorizontalFlip(),
    #     RandomVerticalFlip(),
    #     transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)),
    #     #DEBUG: test divide channel here
    # ])
        test_split = .2
        shuffle_dataset = True
        # random_seed= 42

        dataset = MRDataset(root="./data", output_size=args.img_size)
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(test_split * dataset_size))
        if shuffle_dataset :
            # np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_indices, test_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = DataLoader(dataset, batch_size=args.train_batch_size, num_workers=args.num_workers,
                                  sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers,
                                 sampler=test_sampler)
        return train_loader, test_loader

    else:
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader

import os
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms import RandomRotation, RandomVerticalFlip, RandomHorizontalFlip, Compose, RandomAffine
# from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip, ToTensor, Compose, RandomAffine
from torch.utils.data.sampler import SubsetRandomSampler
from skimage import io, transform

class MRDataset(data.Dataset):
    def __init__(self, root, output_size):
        super().__init__()
        self.root_dir = root
        self.folder_path = self.root_dir + '/data/'
        self.records = pd.read_csv(self.root_dir + '/labels.csv', header=None, names=['id', 'label'])

        self.paths = [self.folder_path + filename +
                      '.npy' for filename in self.records['id'].tolist()[1:]]
        self.labels = [int(x) for x in self.records['label'].tolist()[1:]]

        self.output_size = output_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        array = np.load(self.paths[index])
        label = self.labels[index]
        out_label = None
        if label == 1:
            out_label = np.array([[0, 1, 0, 0]])
        elif label == 0:
            out_label = np.array([[1, 0, 0, 0]])
        elif label == 2:
            out_label = np.array([[0, 0, 1, 0]])
        elif label == 3:
            out_label = np.array([[0, 0, 0, 1]])

        n_slices = array.shape[0]
        out = np.zeros([n_slices, 3, self.output_size, self.output_size])
        for i in range(n_slices):
            out[i] = self._transform_for_one_slice(array[i])
        return torch.from_numpy(out), torch.from_numpy(out_label.astype(float))
    
    def _transform_for_one_slice(self, arr):
        #Resize
        h, w = arr.shape
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        out = transform.resize(arr, (new_h, new_w))
        out = np.repeat(np.expand_dims(out, axis=0), 3, axis=0)
        return out