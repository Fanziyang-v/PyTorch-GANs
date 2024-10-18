"""
Vanilla GAN.

In this implementation, both Generator and Discrminator are defined as MultiLayer Perceptron.

For more details, see: https://arxiv.org/abs/1406.2661
"""

from torch import nn
from torch import Tensor

class Discriminator(nn.Module):
    """
    Disrcminator in GAN.
    
    Model Architecture: [affine - leaky relu - dropout] x 3 - affine - sigmoid
    """
    def __init__(self, image_shape: tuple[int, int, int]) -> None:
        """Initialize Discriminator in GAN.

        Args:
            image_shape(tuple[int, int, int]): shape of image.
        """
        super(Discriminator, self).__init__()
        C, H, W = image_shape
        image_size = C * H * W
        self.model = nn.Sequential(
            nn.Linear(image_size, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, images: Tensor) -> Tensor:
        """Forward pass in Discriminator.

        Args:
            images(Tensor): input images, of shape (N, C, H, W)

        Returns:
            Tensor: probabilities for input images to be real data, of shape (N, 1).
        """
        images = images.view(images.size(0), -1)
        return self.model(images)


class Generator(nn.Module):
    """
    Generator in GAN.

    Model Architecture: [affine - batchnorm - relu] x 4 - affine - tanh
    """
    def __init__(self, image_shape: tuple[int, int, int], latent_dim: int) -> None:
        """Initialize Generator in GAN.

        Args:
            image_shape(tuple[int, int, int]): shape of image.
            latent_dim(int): dimensionality of the latent space.
        """
        super(Generator, self).__init__()
        C, H, W = image_shape
        image_size = C * H * W
        self.image_shape = image_shape
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, image_size), nn.Tanh())
    
    def forward(self, z: Tensor) -> Tensor:
        """Forward pass in Generator.

        Args:
            z(Tensor): latent variables of shape (N, D) that sample from a distribution.

        Returns:
            Tensor: fake images produced by Generator.
        """
        images: Tensor = self.model(z)
        return images.view(-1, *self.image_shape)
