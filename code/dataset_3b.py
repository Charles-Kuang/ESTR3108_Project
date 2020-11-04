import os
import cv2
import torch
import os.path
import torch.utils.data
from pathlib import Path
from skimage import io, util
from torchvision import transforms
from skimage.transform import resize
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.utils as utils
import numpy as np
import crop

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



class LoadDataset3b(torch.utils.data.Dataset):
    train_img_path = Path('../data/task3b/training_data')
    train_gt_path = Path('../data/task3b/training_gt.csv')
    test_img_path = Path('../data/task3b/test_data')
    test_gt_path = Path('../data/task3b/test_gt.csv')
    train_list = pd.read_csv(train_gt_path, header=None).values.tolist()
    test_list = pd.read_csv(test_gt_path, header=None).values.tolist()

    def __init__(self, transform=None, train=True):
        self.transform = transform
        self.train = train

    def __len__(self):
        if self.train:
            return len(self.train_list)
        else:
            return len(self.test_list)

    def __getitem__(self, idx):
        if self.train:
            img_name = self.train_list[idx][0] + '.jpg'
            mask_name = self.train_list[idx][0] + '_Segmentation.png'
            if self.train_list[idx][1] == 'benign':
                self.train_list[idx][1] = torch.tensor(0)
            elif self.train_list[idx][1] == 'malignant':
                self.train_list[idx][1] = torch.tensor(1)
            label = self.train_list[idx][1]
            image = crop.crop(img_name, mask_name, self.train_img_path, idx)
            image = resize(image, (250,250), preserve_range=False, anti_aliasing=True)
            image = util.img_as_ubyte(image)
            image = transform(image)
        else:
            img_name = self.test_list[idx][0] + '.jpg'
            mask_name = self.test_list[idx][0] + '_Segmentation.png'
            label = self.test_list[idx][1]
            image = crop.crop(img_name, mask_name, self.test_img_path, idx)
            #cv2.imshow('1', image)
            #cv2.waitKey(0)
            image = resize(image, (250, 250), preserve_range=False, anti_aliasing=True)
            image = util.img_as_ubyte(image)
            image = transform(image)
        return image, label