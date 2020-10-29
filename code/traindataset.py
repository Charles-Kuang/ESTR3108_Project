import torch.utils.data as data
import torch

from scipy.ndimage import imread
import os
import os.path
import glob

from torchvision import transforms

import numpy as np


def make_dataset(root, train=True):

    dataset = []

    if train:
        dirgt = os.path.join(root, './train_data/groundtruth')
        dirimg = os.path.join(root, './train_data/imgs')

        for fGT in glob.glob(os.path.join(dirgt, '*.jpg'))
            fName = os.path.basename(fGT)
            fImg = 'train_ori' + fName[8:]
            dataset.append([os.path.join(dirimg, fImg), os.path.join(dirgt, fName)])

    return dataset

class MyTrainData(data.Dataset):  # Subclass of Dataset
    def __init__(self, root, transform=None, train=True):
        self.train = train
        if self.train:
            train_set_path = make_dataset(root, train)

    def __getitem__(self, idx):
        if self.train:
            img_path, gt_path = self.train_set_path[idx]

            img = imread(img_path)
            img = np.atleast_3d(img).transpose(2,0,1).astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min())
            img = torch.from_numpy(img).float()

            gt = imread(gt_path)
            gt = np.atleast_3d(gt).transpose(2,0,1)
            gt = gt / 255.0
            gt = torch.from_numpy(gt).float

        return img, gt

    def __len__(self):
        return len(self.train_set_path)
