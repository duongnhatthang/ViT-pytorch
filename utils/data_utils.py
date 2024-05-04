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
        data_path = "./data/"
        # data_path = "/home/thangduong/kneeOA/data/"
        trainset = MRDataset(label_file=data_path+"MOAK20180911_cv0.csv", 
                            src_path=data_path+"MOAKS_study_top20", 
                            out_img_size=args.img_size, dataset=2, is_train = True)
        testset = MRDataset(label_file=data_path+"MOAK20180911_cv0.csv", 
                            src_path=data_path+"MOAKS_study_top20", 
                            out_img_size=args.img_size, dataset=2, is_train = False)
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

import numpy as np
import torch.utils.data as data
from torchvision import transforms
from skimage import io, transform

class MRDataset(data.Dataset):
    def __init__(self, label_file, src_path, out_img_size=224, dataset=0, mode='multiclass', folder_num=5, is_train = True):
        """
        folder_num is the cross-validation split, temporarily keep here from the legacy code
        """
        super().__init__()
        self.out_img_size = out_img_size
        self.output_size = 4
        if mode.startswith('binary'):
            self.output_size = 2
        self.rotate_degree = 1
        self.shape = (out_img_size, out_img_size)
        self.is_train = is_train
        self.dataset = dataset
        self.length = out_img_size * out_img_size
        label_data, image_data = self.import_data(label_file, src_path, dataset, mode, folder_num)
        self.process_data(label_data, image_data)
        
    def import_data(self, label_file, src_path, dataset=0, mode='multiclass', folder_num=5):
        dataset_opt = {
            0: 'top10',
            1: 'top15',
            2: 'top20'
        }

        # (id_L_M, awld_data, label)
        label_data = [(list(), list(), list()) for x in range(folder_num)]
        image_data = dict()
        cnt = -1
        sample_cnt = 0
        with open(label_file, "r") as fp:
            for line in fp:
                cnt += 1
                # ignore header
                if cnt == 0:
                    continue
                if cnt % 1000 == 0:
                    logger.info('cnt={}'.format(cnt))
                    
                # if cnt % 150 == 0 and cnt > 1:
                #     logger.info('cnt={}'.format(cnt))
                #     break # TODO: for quick debug
                
                fields = line.replace('\n', '').split(",")
                # if fields[0] != dataset_opt[dataset]:
                #    continue
                group = int(fields[7])
                if mode == 'multiclass' and group not in (0, 1, 2, 3):
                    continue
                if mode == 'binary' and group not in (0, 1, 2, 4):
                    continue
                if mode == 'binary2' and group not in (0, 1, 2, 5):
                    continue

                sample_cnt += 1

                cv = int(fields[6])
                train_data = label_data[cv][0]
                validation_data = label_data[cv][1]
                test_data = label_data[cv][2]

                label = int(fields[4])

                key = '{}_{}_{}'.format(fields[1], fields[2], fields[3])
                if group == 0 or group >= 3:
                    train_data.append((key, label))
                elif group == 1:
                    validation_data.append((key, label))
                elif group == 2:
                    test_data.append((key, label))

                # image exists
                if key in image_data.keys():
                    continue

                # load image data
                awld_file = '{}/{}_axld.npz'.format(src_path, key)
                load_data = np.load(awld_file)
                data_list = [np.reshape(x[1][int(x[1].shape[0] / 2) - self.out_img_size//2:int(x[1].shape[0] / 2) + self.out_img_size//2,
                                        int(x[1].shape[1] / 2) - self.out_img_size//2:int(x[1].shape[1] / 2) + self.out_img_size//2], self.length) for x in
                            load_data.items()]

                awld_data = np.asmatrix(data_list, dtype=np.float16)
                pv = np.percentile(data_list, 99) * 2
                # pv = np.percentile(awld_data, 99) * 2 #TODO: quick-fix here
                awld_data[awld_data > pv] = 0

                std = np.std(awld_data, dtype=np.float64)
                mean = np.mean(awld_data, dtype=np.float64)
                awld_data_std = (awld_data - mean) / std

                image_data[key] = awld_data_std
        return label_data, image_data
    
    def process_data(self, label_data, image_data):
        full_label_data, image_data = label_data[0], image_data #label_data[0] = 1st cross-validation set
        if self.is_train:
            label_list = full_label_data[0]
        else:
            label_list = full_label_data[1]

        self.labels = []
        self.imgs = []

        for key, label in label_list:
            if key not in image_data:
                continue
            axld_std = image_data[key]

            r = self.rotate_degree % 4
            axld_list = [np.reshape(np.rot90(np.reshape(x, newshape=self.shape), r), self.length) for x in axld_std.tolist()]
            axld_data = np.asmatrix(axld_list, dtype=np.float16)

            # train_data.append((axld_data, label))
            self.labels.append(label)
            self.imgs.append(axld_data)
            
            r += 1# Useless?

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        out_label = None
        if label == 1:
            out_label = np.array([0, 1, 0, 0])
        elif label == 0:
            out_label = np.array([1, 0, 0, 0])
        elif label == 2:
            out_label = np.array([0, 0, 1, 0])
        elif label == 3:
            out_label = np.array([0, 0, 0, 1])

        if self.dataset == 0:
            n_slices = 10
        elif self.dataset == 1:
            n_slices = 15
        else:
            n_slices = 20
        out = np.zeros([n_slices, 3, self.out_img_size, self.out_img_size])
        for i in range(n_slices):
            out[i] = self._transform_for_one_slice(self.imgs[index][i], i)
        return torch.from_numpy(out), torch.from_numpy(out_label.astype(float))
    
    def _transform_for_one_slice(self, arr, slice_idx):
        #Resize
        # h, w = arr.shape
        h, w = 384, 384
        if isinstance(self.out_img_size, int):
            if h > w:
                new_h, new_w = self.out_img_size * h / w, self.out_img_size
            else:
                new_h, new_w = self.out_img_size, self.out_img_size * w / h
        else:
            new_h, new_w = self.out_img_size

        new_h, new_w = int(new_h), int(new_w)

        out = transform.resize(arr, (new_h, new_w))
        out = np.repeat(np.expand_dims(out, axis=0), 3, axis=0)
        return out
        # return (out-self.mean[slice_idx])/self.std[slice_idx]


# import os
# import pandas as pd
# import torch
# import torch.nn.functional as F
# import torchvision.transforms.functional as TF
# from torchvision.transforms import RandomRotation, RandomVerticalFlip, RandomHorizontalFlip, Compose, RandomAffine
# # from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip, ToTensor, Compose, RandomAffine
# from torch.utils.data.sampler import SubsetRandomSampler

# class MRDataset(data.Dataset):
#     def __init__(self, root, output_size, mean = None, std = None):
#         super().__init__()
#         self.root_dir = root
#         self.folder_path = self.root_dir + '/data/'
#         self.records = pd.read_csv(self.root_dir + '/labels.csv', header=None, names=['id', 'label'])

#         self.paths = [self.folder_path + filename +
#                       '.npy' for filename in self.records['id'].tolist()[1:]]
#         self.labels = [int(x) for x in self.records['label'].tolist()[1:]]

#         self.output_size = output_size
#         if mean is not None:
#             self.mean = mean
#         else:
#             self.mean = [52.71059247, 53.63070621, 54.42507352, 54.9882445,  55.24719491, 55.25069302,
#  55.05574835, 54.7266365,  54.28619861, 53.81264898, 53.34875302, 52.91868226,
#  52.5478438,  52.25599246, 52.02444738, 51.87840007, 51.81854203, 51.82532474,
#  51.89948723, 52.02127323]
#         if std is not None:
#             self.std = std
#         else:
#             self.std = [10.29691447, 10.5882155,  10.78170394, 10.88856715, 10.94099957, 10.90728367,
#  10.82318519, 10.69989101, 10.53939241, 10.359683,   10.13378993,  9.93059632,
#   9.75945511,  9.61362473,  9.47307353,  9.36288117,  9.29843515,  9.28210771,
#   9.27132131,  9.25777826]

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, index):
#         array = np.load(self.paths[index])
#         label = self.labels[index]
#         out_label = None
#         if label == 1:
#             out_label = np.array([[0, 1, 0, 0]])
#         elif label == 0:
#             out_label = np.array([[1, 0, 0, 0]])
#         elif label == 2:
#             out_label = np.array([[0, 0, 1, 0]])
#         elif label == 3:
#             out_label = np.array([[0, 0, 0, 1]])

#         n_slices = array.shape[0]
#         out = np.zeros([n_slices, 3, self.output_size, self.output_size])
#         for i in range(n_slices):
#             out[i] = self._transform_for_one_slice(array[i], i)
#         return torch.from_numpy(out), torch.from_numpy(out_label.astype(float))
    
#     def _transform_for_one_slice(self, arr, slice_idx):
#         #Resize
#         h, w = arr.shape
#         if isinstance(self.output_size, int):
#             if h > w:
#                 new_h, new_w = self.output_size * h / w, self.output_size
#             else:
#                 new_h, new_w = self.output_size, self.output_size * w / h
#         else:
#             new_h, new_w = self.output_size

#         new_h, new_w = int(new_h), int(new_w)

#         out = transform.resize(arr, (new_h, new_w))
#         out = np.repeat(np.expand_dims(out, axis=0), 3, axis=0)
#         return (out-self.mean[slice_idx])/self.std[slice_idx]