import os
import pandas as pd
import torch
import torch.utils.data
from skimage import io
from skimage import color
from torchvision import transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class LoadDataset1(torch.utils.data.Dataset):
    train_img_path = '../data/task1/training_data'
    train_mask_img_path = '../data/task1/training_gt'
    test_img_path = '../data/task1/test_data'
    test_mask_img_path = '../data/task1/test_gt'

    def __init__(self, transform=None, train=True):
        self.transform = transform
        self.train = train
        if self.train:
            self.mask_img = os.listdir(self.train_mask_img_path)
        else:
            self.mask_img = os.listdir(self.test_mask_img_path)

    def __len__(self):
        return len(self.mask_img)

    def __getitem__(self, idx):
        mask_image_name = self.mask_img[idx]
        origin_image_name = mask_image_name[:12] + '.jpg'
        if self.train:
            origin_image = io.imread(os.path.join(self.train_img_path, origin_image_name))
            mask_image = io.imread(os.path.join(self.train_mask_img_path, mask_image_name))
            mask_image = color.gray2rgb(mask_image)
            origin_image = transform(origin_image)
            mask_image = transform(mask_image)
        else:
            origin_image = io.imread(os.path.join(self.test_img_path, origin_image_name))
            mask_image = io.imread(os.path.join(self.test_mask_img_path, mask_image_name))
            mask_image = color.gray2rgb(mask_image)
            origin_image = transform(origin_image)
            mask_image = transform(mask_image)
        return origin_image, mask_image

test = LoadDataset1()
print(test[1])