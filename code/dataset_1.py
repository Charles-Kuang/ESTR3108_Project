import os
import pandas as pd
import torch
import torch.utils.data
from skimage import io
from torchvision import transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

