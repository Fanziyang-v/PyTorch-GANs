from torch import nn
from torch import Tensor

class ResNet(nn.Module):
    """ResNet for CIFAR-10."""
    def __init__(self, num_channels: int, num_classes: int=10, layers: list[int]=[3, 3, 3]) -> None:
        """Initialize ResNet.

        Args:
            layers(list): specify number of residual blocks in each layer. Default is [3, 3, 3].
        """
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(num_channels, 16)
        # self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self.__make_layer(16, layers[0])
        self.layer2 = self.__make_layer(32, layers[1], 2)
        self.layer3 = self.__make_layer(64, layers[2], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def __make_layer(self, out_channels: int, blocks: int, stride=1) -> nn.Module:
        """Construct ResNet layer.

        Args:
            out_channels(int): number of output channels in this layer.
            blocks(int): number of residual block in this layer.
            stride(int): stride of convolution.

        Returns:
            Module: ResNet layer.
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = [ResidualBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, images: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            images(Tensor): input images of shape (N, C, 32, 32)
        
        Returns:
            Tensor: scores matrix of shape (N, D)
        """
        out: Tensor = self.conv(images)
        # out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, in_channels: int, out_channels: int, stride: int=1, downsample=None) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.downsample = downsample
    
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu2(out)
        return out


# 3x3 convolution.
def conv3x3(in_channels: int, out_channels: int, stride: int=1) -> nn.Module:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1)
