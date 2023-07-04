import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob


class DataLoader_train_pre(Dataset):
    def __init__(self, data_dir, label_dir):
        super(DataLoader_train_pre, self).__init__()
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.train_fns_100 = sorted(glob.glob(
            os.path.join(data_dir, '100kv/*.npy')))[:1000]
        self.train_fns_140 = sorted(glob.glob(os.path.join(data_dir, '140kv/*.npy')))[:1000]
        self.label_fns = sorted(glob.glob(os.path.join(label_dir, '*.npy')))
        print('fetch {} samples for training/testing'.format(len(self.train_fns_100)))
        print('fetch {} samples for label'.format(len(self.label_fns)))

    def __getitem__(self, index):
        # fetch image
        if len(self.label_fns) != len(self.train_fns_100):
            raise RuntimeError("Unequal number between 100kv files and 140kv files!")

        fn_100 = self.train_fns_100[index]
        fn_140 = self.train_fns_140[index]
        fn_label = self.label_fns[index]

        label_each = np.load(fn_label)
        im_100 = np.load(fn_100)
        im_140 = np.load(fn_140)

        im_100 = np.array(im_100, dtype=np.float32)
        im_140 = np.array(im_140, dtype=np.float32)
        im = np.stack((im_100, im_140), axis=0)
        im = torch.from_numpy(im)

        label_each = np.array(label_each, dtype=np.float32)
        label_each = torch.from_numpy(label_each)

        return im, label_each

    def __len__(self):
        return len(self.train_fns_100)


class DataLoader_test_pre(Dataset):
    def __init__(self, data_dir):
        super(DataLoader_test_pre, self).__init__()
        self.data_dir = data_dir
        self.train_fns_100 = sorted(glob.glob(
            os.path.join(data_dir, '100kv/*.npy')))[:2]
        self.train_fns_140 = sorted(glob.glob(os.path.join(data_dir, '140kv/*.npy')))[:2]
        print('fetch {} samples for training/testing'.format(len(self.train_fns_100)))

    def __getitem__(self, index):
        # fetch image
        fn_100 = self.train_fns_100[index]
        fn_140 = self.train_fns_140[index]

        im_100 = np.load(fn_100)
        im_140 = np.load(fn_140)

        im_100 = np.array(im_100, dtype=np.float32)
        im_140 = np.array(im_140, dtype=np.float32)
        im = np.stack((im_100, im_140), axis=0)
        im = torch.from_numpy(im)
        return im, index

    def __len__(self):
        return len(self.train_fns_100)


class DataLoader_second(Dataset):
    def __init__(self, data_dir):
        super(DataLoader_second, self).__init__()
        self.data_dir = data_dir
        self.train_fns_100 = sorted(glob.glob(
            os.path.join(data_dir, '100kv/*.npy')))
        self.train_fns_140 = sorted(glob.glob(os.path.join(data_dir, '140kv/*.npy')))
        print('fetch {} samples for training/testing'.format(len(self.train_fns_100)))

    def __getitem__(self, index):
        # fetch image
        fn_100 = self.train_fns_100[index]
        fn_140 = self.train_fns_140[index]

        im_100 = np.load(fn_100)
        im_140 = np.load(fn_140)

        im_100 = np.array(im_100, dtype=np.float32)
        im_140 = np.array(im_140, dtype=np.float32)

        im = np.stack((im_100, im_140), axis=0)
        im = torch.from_numpy(im)
        return im

    def __len__(self):
        return len(self.train_fns_100)


class DataLoader_SECT(Dataset):
    def __init__(self, data_dir):
        super(DataLoader_SECT, self).__init__()
        self.data_dir = data_dir
        self.train_fns_100 = sorted(glob.glob(
            os.path.join(data_dir, '100kv/*.npy')))
        self.train_fns_140 = sorted(glob.glob(os.path.join(data_dir, '140kv/*.npy')))
        print('fetch {} samples for training/testing'.format(len(self.train_fns_100)))

    def __getitem__(self, index):
        # fetch image
        if len(self.train_fns_140) != len(self.train_fns_100):
            raise RuntimeError("Unequal number between 100kv files and 140kv files!")

        fn_100 = self.train_fns_100[index]
        fn_140 = self.train_fns_140[index]

        im_100 = np.load(fn_100)
        im_140 = np.load(fn_140)

        im_100 = np.array(im_100, dtype=np.float32)
        im_140 = np.array(im_140, dtype=np.float32)

        im = np.stack((im_100, im_140), axis=0)
        im = torch.from_numpy(im)
        return im

    def __len__(self):
        return len(self.train_fns_100)

