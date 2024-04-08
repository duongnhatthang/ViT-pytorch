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
    def __init__(self, root, output_size, mean = None, std = None):
        super().__init__()
        self.root_dir = root
        self.folder_path = self.root_dir + '/data/'
        self.records = pd.read_csv(self.root_dir + '/labels.csv', header=None, names=['id', 'label'])

        self.paths = [self.folder_path + filename +
                      '.npy' for filename in self.records['id'].tolist()[1:]]
        self.labels = [int(x) for x in self.records['label'].tolist()[1:]]

        self.output_size = output_size
        if mean is not None:
            self.mean = mean
        else:
            self.mean = [52.71059247, 53.63070621, 54.42507352, 54.9882445,  55.24719491, 55.25069302,
 55.05574835, 54.7266365,  54.28619861, 53.81264898, 53.34875302, 52.91868226,
 52.5478438,  52.25599246, 52.02444738, 51.87840007, 51.81854203, 51.82532474,
 51.89948723, 52.02127323]
        if std is not None:
            self.std = std
        else:
            self.std = [10.29691447, 10.5882155,  10.78170394, 10.88856715, 10.94099957, 10.90728367,
 10.82318519, 10.69989101, 10.53939241, 10.359683,   10.13378993,  9.93059632,
  9.75945511,  9.61362473,  9.47307353,  9.36288117,  9.29843515,  9.28210771,
  9.27132131,  9.25777826]

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
            out[i] = self._transform_for_one_slice(array[i], i)
        return torch.from_numpy(out), torch.from_numpy(out_label.astype(float))
    
    def _transform_for_one_slice(self, arr, slice_idx):
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
        return (out-self.mean[slice_idx])/self.std[slice_idx]