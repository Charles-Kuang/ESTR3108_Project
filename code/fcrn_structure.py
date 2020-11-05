import torch
import torch.nn as nn
from torch.utils import model_zoo
from torchsummary import summary
import torch.optim as optim
import torchvision.models as models

__all__ = ['FCResNet', 'FCResnet50']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}  #

class BilinearConvTranspose2d(nn.ConvTranspose2d):
    """A conv transpose initialized to bilinear interpolation."""

    def __init__(self, channels, stride, groups=1):
        """Set up the layer.
        Parameters
        ----------
        channels: int
            The number of input and output channels
        stride: int or tuple
            The amount of upsampling to do
        groups: int
            Set to 1 for a standard convolution. Set equal to channels to
            make sure there is no cross-talk between channels.
        """
        if isinstance(stride, int):
            stride = (stride, stride)

        assert groups in (1, channels), "Must use no grouping, " + \
            "or one group per channel"

        kernel_size = (2 * stride[0] - 1, 2 * stride[1] - 1)
        padding = (stride[0] - 1, stride[1] - 1)
        super().__init__(
            channels, channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            output_padding=1)

    def reset_parameters(self):
        """Reset the weight and bias."""
        nn.init.constant(self.bias, 0)
        nn.init.constant(self.weight, 0)
        bilinear_kernel = self.bilinear_kernel(self.stride)
        for i in range(self.in_channels):
            if self.groups == 1:
                j = i
            else:
                j = 0
            self.weight.data[i, j] = bilinear_kernel

    @staticmethod
    def bilinear_kernel(stride):
        """Generate a bilinear upsampling kernel."""
        num_dims = len(stride)

        shape = (1,) * num_dims
        bilinear_kernel = torch.ones(*shape)

        # The bilinear kernel is separable in its spatial dimensions
        # Build up the kernel channel by channel
        for channel in range(num_dims):
            channel_stride = stride[channel]
            kernel_size = 2 * channel_stride - 1
            # e.g. with stride = 4
            # delta = [-3, -2, -1, 0, 1, 2, 3]
            # channel_filter = [0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
            delta = torch.arange(1 - channel_stride, channel_stride)
            channel_filter = (1 - torch.abs(delta / channel_stride))
            # Apply the channel filter to the current channel
            shape = [1] * num_dims
            shape[channel] = kernel_size
            bilinear_kernel = bilinear_kernel * channel_filter.view(shape)
        return bilinear_kernel

# define 3x3 convolutional layer
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# define residual block
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.conv3 = nn.Conv2d(outplanes, outplanes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ResNet
class FCResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(FCResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.fusion1 = nn.Conv2d(128 * block.expansion, 2, kernel_size=1, bias=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.fusion2 = nn.Conv2d(256 * block.expansion, 2, kernel_size=1, bias=False)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fusion3 = nn.Conv2d(512 * block.expansion, 2, kernel_size=1, bias=False)
        self.deconv1 = BilinearConvTranspose2d(2, 2)
        self.deconv2 = BilinearConvTranspose2d(2, 2)
        self.deconv3 = BilinearConvTranspose2d(2, 8)

        for m in self.modules():  # modules: store all the layers in self
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # initializa the weight/kernel
            elif isinstance(m, nn.BatchNorm2d):  # mean->0, variance->1
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        fuse_1 = self.fusion1(x)
        x = self.layer3(x)
        fuse_2 = self.fusion2(x)
        x = self.layer4(x)
        x = self.fusion3(x)
        x = self.deconv1(x)
        x = self.deconv2(x + fuse_2)
        x = self.deconv3(x + fuse_1)
        return x


def FCResnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FCResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        resnet50 = models.resnet50(pretrained=True)
        pretrained_dict = resnet50.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model