import os
import pandas as pd
import torch
import torch.utils.data
from skimage import io
from torchvision import transforms
from pathlib import Path

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class LoadDataset3(torch.utils.data.Dataset):
    train_img_path = Path('../data/task3/training_data')
    train_gt_path = Path('../data/task3/training_gt.csv')
    test_img_path = Path('../data/task3/test_data')
    test_gt_path = Path('../data/task3/test_gt.csv')
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
            image, label = io.imread(os.path.join(self.train_img_path, img_name)), self.train_list[idx][1]
            image = transform(image)
        else:
            img_name = self.test_list[idx][0] + '.jpg'
            image, label = io.imread(os.path.join(self.test_img_path, img_name)), self.test_list[idx][1]
            image = transform(image)
        return image, label

test = LoadDataset3()
print(test[0])
