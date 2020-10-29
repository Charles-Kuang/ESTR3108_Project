import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo#

__all__ = ['ResNet', 'resnet18']

model_urls = {
     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}#

#define 3x3 convolutional layer
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

#define residual block
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)