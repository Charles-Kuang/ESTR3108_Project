import os
import torch
import torch.utils.data
from skimage import color
import torch.utils.data
from skimage import io, util
from skimage.transform import resize
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.utils as utils
from preprocessing import processing
import numpy as np

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform4gray = transforms.Compose(
    [transforms.ToTensor()])


class LoadDataset1(torch.utils.data.Dataset):
    train_img_path = Path('../data/task1/training_data')
    train_mask_img_path = Path('../data/task1/training_gt')
    test_img_path = Path('../data/task1/test_data')
    test_mask_img_path = Path('../data/task1/test_gt')

    def __init__(self, transform=None, transform4gray=None, train=True):
        self.transform = transform
        self.transform4gray = transform4gray
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
            origin_image = resize(origin_image, (250, 250), preserve_range=False, anti_aliasing=False)
            origin_image = util.img_as_ubyte(origin_image)
            origin_image = processing(origin_image)
            mask_image = resize(mask_image, (250, 250), preserve_range=False, anti_aliasing=False)
            mask_image = util.img_as_ubyte(mask_image)
            origin_image = transform(origin_image)
            mask_image = transform4gray(mask_image)
        else:
            origin_image = io.imread(os.path.join(self.test_img_path, origin_image_name))
            mask_image = io.imread(os.path.join(self.test_mask_img_path, mask_image_name))
            origin_image = resize(origin_image, (250, 250), preserve_range=False, anti_aliasing=False)
            origin_image = util.img_as_ubyte(origin_image)
            #origin_image = processing(origin_image)
            mask_image = resize(mask_image, (250, 250), preserve_range=False, anti_aliasing=False)
            mask_image = util.img_as_ubyte(mask_image)
            origin_image = transform(origin_image)
            mask_image = transform4gray(mask_image)
        return origin_image, mask_image

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

trainset = LoadDataset1(transform=transform, transform4gray=transform4gray, train=True)
origin, mask = trainset[89]
imshow(utils.make_grid(origin))