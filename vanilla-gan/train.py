import os
from argparse import Namespace, ArgumentParser
import torch
from torch import nn, Tensor
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from gan import Generator, Discriminator

# Image processing.
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5))])

transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# Device configuration.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def denormalize(x: Tensor) -> Tensor:
    out = (x + 1) / 2
    return out.clamp(0, 1)


def get_args() -> Namespace:
    """Get commandline arguments."""
    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate for Adam optimizer')
    parser.add_argument('--beta1', type=float, default=0.5, help='first momentum term for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='second momentum term for Adam')
    parser.add_argument('--batch_size', type=int, default=64, help='size of a mini-batch')
    parser.add_argument('--num_epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--dataset', type=str, default='MNIST', help='training dataset(MNIST | FashionMNIST | CIFAR10)')
    parser.add_argument('--sample_dir', type=str, default='samples', help='directory of image samples')
    parser.add_argument('--interval', type=int, default=1, help='epoch interval between image samples')
    parser.add_argument('--logdir', type=str, default='runs', help='directory of running log')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='directory for saving model checkpoints')
    parser.add_argument('--seed', type=str, default=10213, help='random seed')
    return parser.parse_args()


def setup(args: Namespace) -> None:
    torch.manual_seed(args.seed)
    # Create directory if not exists.
    if not os.path.exists(os.path.join(args.sample_dir, args.dataset)):
        os.makedirs(os.path.join(args.sample_dir, args.dataset))
    if not os.path.exists(os.path.join(args.ckpt_dir, args.dataset)):
        os.makedirs(os.path.join(args.ckpt_dir, args.dataset))


def get_data_loader(args: Namespace) -> DataLoader:
    """Get data loader."""
    if args.dataset == 'MNIST':
        data = datasets.MNIST(root='../data', train=True, download=True, transform=transform_mnist)
    elif args.dataset == 'FashionMNIST':
        data = datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform_mnist)
    elif args.dataset == 'CIFAR10':
        data = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_cifar)
    else:
        raise ValueError(f'Unkown dataset: {args.dataset}, support dataset: MNIST | FashionMNIST | CIFAR10')
    return DataLoader(dataset=data, batch_size=args.batch_size, num_workers=4, shuffle=True)


def train(args: Namespace, 
          G: Generator, D: Discriminator, 
          data_loader: DataLoader) -> None:
    """Train Generator and Discriminator.

    Args:
        args(Namespace): arguments.
        G(Generator): Generator in GAN.
        D(Discriminator): Discriminator in GAN.
    """
    writer = SummaryWriter(os.path.join(args.logdir, args.dataset))

    # generate fixed noise for sampling.
    fixed_noise = torch.rand(64, args.latent_dim).to(device)

    # Loss and optimizer.
    criterion = nn.BCELoss().to(device)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Start training.
    for epoch in range(args.num_epochs):
        total_d_loss = total_g_loss = 0
        for images, _ in data_loader:
            m = images.size(0)
            images: Tensor = images.to(device)
            images = images.view(m, -1)
            # Create real and fake labels.
            real_labels = torch.ones(m, 1).to(device)
            fake_labels = torch.zeros(m, 1).to(device)
            # ================================================================== #
            #                      Train the discriminator                       #
            # ================================================================== #

            # Forward pass
            outputs = D(images)
            d_loss_real: Tensor = criterion(outputs, real_labels)
            
            z = torch.rand(m, args.latent_dim).to(device)
            fake_images: Tensor = G(z).detach()
            outputs = D(fake_images)
            d_loss_fake: Tensor = criterion(outputs, fake_labels)

            # Backward pass
            d_loss: Tensor = d_loss_real + d_loss_fake
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()
            total_d_loss += d_loss

            # ================================================================== #
            #                        Train the generator                         #
            # ================================================================== #
            
            # Forward pass
            z = torch.rand(images.size(0), args.latent_dim).to(device)
            fake_images: Tensor = G(z)
            outputs = D(fake_images)
            
            # Backward pass
            g_loss: Tensor = criterion(outputs, real_labels)
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()
            total_g_loss += g_loss
        print(f'''
=====================================
Epoch: [{epoch + 1}/{args.num_epochs}]
Discriminator Loss: {total_d_loss / len(data_loader):.4f}
Generator Loss: {total_g_loss / len(data_loader):.4f}
=====================================''')
        # Log Discriminator and Generator loss.
        writer.add_scalar('Discriminator Loss', total_d_loss / len(data_loader), epoch + 1)
        writer.add_scalar('Generator Loss', total_g_loss / len(data_loader), epoch + 1)
        fake_images: Tensor = G(fixed_noise)
        img_grid = make_grid(denormalize(fake_images), nrow=8, padding=2)
        writer.add_image('Fake Images', img_grid, epoch + 1)
        if (epoch + 1) % args.interval == 0:
            save_image(img_grid, os.path.join(args.sample_dir, args.dataset, f'fake_images_{epoch + 1}.png'))
    # Save the model checkpoints.
    torch.save(G.state_dict(), os.path.join(args.ckpt_dir, args.dataset, 'G.ckpt'))
    torch.save(D.state_dict(), os.path.join(args.ckpt_dir, args.dataset, 'D.ckpt'))


def main() -> None:
    args = get_args()
    setup(args)
    image_shape = (1, 28, 28) if args.dataset in ('MNIST', 'FashionMNIST') else (3, 32, 32)
    data_loader = get_data_loader(args)
    # Generator and Discrminator.
    G = Generator(image_shape=image_shape, latent_dim=args.latent_dim).to(device)
    D = Discriminator(image_shape=image_shape).to(device)
    train(args, G, D, data_loader)


if __name__ == '__main__':
    main()
