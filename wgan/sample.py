import os
from argparse import Namespace, ArgumentParser
import torch
from torch import Tensor
from torchvision.utils import save_image, make_grid
from wgan import Generator

# Device configuration.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denormalize(x: Tensor) -> Tensor:
    out = (x + 1) / 2
    return out.clamp(0, 1)


def get_args() -> Namespace:
    """Get commandline arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "--latent_dim", type=int, default=100, help="dimensionality of the latent space"
    )
    parser.add_argument(
        "--sample_dir", type=str, default="samples", help="directory of image samples"
    )
    parser.add_argument(
        "--num_sample", type=int, default=5, help="number of image samples"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        help="training dataset(MNIST | FashionMNIST | CIFAR10)",
    )
    parser.add_argument("--g_ckpt", type=str, help="generator model checkpoint")
    parser.add_argument("--seed", type=int, default=10213, help="random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    C = 1 if args.dataset in ("MNIST", "FashionMNIST") else 3
    image_size = 32 if args.dataset in ("MNIST", "FashionMNIST", "CIFAR10") else 64
    torch.manual_seed(args.seed)
    G = Generator(num_channels=C, image_size=image_size, latent_dim=args.latent_dim).to(
        device
    )
    G.load_state_dict(torch.load(args.g_ckpt, weights_only=True))
    for i in range(args.num_sample):
        noise = torch.rand(64, args.latent_dim).to(device)
        fake_images = G(noise)
        img_grid = make_grid(denormalize(fake_images), nrow=8, padding=2)
        save_image(
            img_grid,
            os.path.join(args.sample_dir, args.dataset, f"fake_images_{i + 1}.png"),
        )
