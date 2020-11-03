import os
import pandas as pd
import torch
import torch.utils.data
from skimage import io, util
from skimage.transform import resize
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.utils as utils

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
            if self.train_list[idx][1] == 'benign':
                self.train_list[idx][1] = torch.tensor(0)
            elif self.train_list[idx][1] == 'malignant':
                self.train_list[idx][1] = torch.tensor(1)
            image, label = io.imread(os.path.join(self.train_img_path, img_name)), self.train_list[idx][1]
            image = resize(image, (250,250), preserve_range=False, anti_aliasing=False)
            image = util.img_as_ubyte(image)
            image = transform(image)
        else:
            img_name = self.test_list[idx][0] + '.jpg'
            image, label = io.imread(os.path.join(self.test_img_path, img_name)), self.test_list[idx][1]
            image = resize(image, (250, 250), preserve_range=False, anti_aliasing=False)
            image = util.img_as_ubyte(image)
            image = transform(image)
        return image, label

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

#test = LoadDataset3()
#print(test[3])

"""
trans = transforms.Compose([transforms.ToTensor()])
test = LoadDataset3()
x, y = test[10]
imshow(x)
plt.show()
print (y)
"""