"""
U-Net for Generator.

For more details, see: http://arxiv.org/abs/1505.04597
"""

import torch
from torch import nn, Tensor
from torch.nn import functional as F

class UNet(nn.Module):
    """U-Net."""
    def __init__(self, num_channels: int) -> None:
        """Initialize U-Net.
        
        Args:
            num_channels(int): number of channels.
        """
        super(UNet, self).__init__()
        self.in_conv = DoubleConv(num_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024)
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        self.out_conv = nn.Conv2d(64, num_channels, kernel_size=1, stride=1)
        self.tanh = nn.Tanh()

    def forward(self, z: Tensor) -> Tensor:
        """Forward pass in U-Net.
        
        Args:
            z(Tensor): latent variables of shape (N, C, H, W). 
        
        Returns:
            Tensor: fake images of shape (N, C, H, W) produced by generator.
        """
        # go down
        x1 = self.in_conv(z)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        out = self.down4(x4)
        # go up
        out = self.up1(out, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.out_conv(out)
        out = self.tanh(out)
        return out
        

class UpBlock(nn.Module):
    """UpBlock in U-Net."""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize a UpBlock in U-Net.
        
        Args:
            in_channels(int): number of channels of input feature map.
            out_channels(int): number of channels of output feature map.
        """
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DownBlock(nn.Module):
    """DownBlock in U-Net."""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize a DownBlock in U-Net.
        
        Args:
            in_channels(int): number of channels of input feature map.
            out_channels(int): number of channels of output feature map.
        """
        super(DownBlock, self).__init__()
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.down(x)
        out = self.conv(out)
        return out


class DoubleConv(nn.Module):
    """DoubleConv block.
    
    Architecutre: [conv - batchnorm - relu] x 2
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU())
    
    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x)
