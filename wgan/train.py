import os
from argparse import Namespace, ArgumentParser
import torch
from torch import nn, Tensor
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from wgan import Generator, Discriminator

# Image processing.
transform_mnist = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5)),
    ]
)

transform_cifar = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

# Device configuration.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denormalize(x: Tensor) -> Tensor:
    out = (x + 1) / 2
    return out.clamp(0, 1)


# mini-batch generator
def generate(data_loader: DataLoader):
    while True:
        for images, _ in data_loader:
            # Reach the last batch without enough images, just break
            if data_loader.batch_size != len(images):
                break
            yield images


def get_args() -> Namespace:
    """Get commandline arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "--lr", type=float, default=0.00005, help="learning rate for RMSProp optimizer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="size of a mini-batch"
    )
    parser.add_argument("--gen_iters", type=int, default=100000, help="training epochs")
    parser.add_argument(
        "--latent_dim", type=int, default=100, help="dimensionality of the latent space"
    )
    parser.add_argument(
        "--n_critic",
        type=int,
        default=5,
        help="number of training iterations in discriminator per generator training iteration",
    )
    parser.add_argument(
        "--clip_limit",
        type=float,
        default=0.01,
        help="clipping parameter for weight clipping",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        help="training dataset(MNIST | FashionMNIST | CIFAR10)",
    )
    parser.add_argument(
        "--sample_dir", type=str, default="samples", help="directory of image samples"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1000,
        help="epoch interval between image samples",
    )
    parser.add_argument(
        "--logdir", type=str, default="runs", help="directory of running log"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="checkpoints",
        help="directory for saving model checkpoints",
    )
    parser.add_argument("--seed", type=str, default=10213, help="random seed")
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
    if args.dataset == "MNIST":
        data = datasets.MNIST(
            root="../data", train=True, download=True, transform=transform_mnist
        )
    elif args.dataset == "FashionMNIST":
        data = datasets.FashionMNIST(
            root="../data", train=True, download=True, transform=transform_mnist
        )
    elif args.dataset == "CIFAR10":
        data = datasets.CIFAR10(
            root="../data", train=True, download=True, transform=transform_cifar
        )
    else:
        raise ValueError(
            f"Unkown dataset: {args.dataset}, support dataset: MNIST | FashionMNIST | CIFAR10"
        )
    return DataLoader(
        dataset=data, batch_size=args.batch_size, num_workers=4, shuffle=True
    )


def train(
    args: Namespace, G: Generator, D: Discriminator, data_loader: DataLoader
) -> None:
    """Train Generator and Discriminator.

    Args:
        args(Namespace): arguments.
        G(Generator): Generator in GAN.
        D(Discriminator): Discriminator in GAN.
    """
    gen = generate(data_loader)
    writer = SummaryWriter(os.path.join(args.logdir, args.dataset))

    # generate fixed noise for sampling.
    fixed_noise = torch.rand(64, args.latent_dim).to(device)

    # Loss and optimizer.
    optimizer_G = torch.optim.RMSprop(G.parameters(), lr=args.lr)
    optimizer_D = torch.optim.RMSprop(D.parameters(), lr=args.lr)

    for i in range(args.gen_iters):
        total_d_loss = 0
        for _ in range(args.n_critic):
            images = next(gen)
            images: Tensor = images.to(device)
            # ================================================================== #
            #                      Train the discriminator                       #
            # ================================================================== #

            # Forward pass
            z = torch.rand(args.batch_size, args.latent_dim).to(device)
            real_score: Tensor = D(images)
            fake_score: Tensor = D(G(z))
            d_loss = (fake_score - real_score).mean()

            # Backward pass
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()
            total_d_loss += d_loss

            # Clip weight in [-c, c].
            for params in D.parameters():
                params.data.clamp_(min=-args.clip_limit, max=args.clip_limit)
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Forward pass
        z = torch.rand(args.batch_size, args.latent_dim).to(device)
        fake_score: Tensor = D(G(z))
        g_loss = (-fake_score).mean()

        # Backward pass
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        print(
            f"""
=====================================
Step: [{i + 1}/{args.gen_iters}]
Discriminator Loss: {total_d_loss / args.n_critic:.4f}
Generator Loss: {g_loss:.4f}
====================================="""
        )
        # Log Discriminator and Generator loss.
        writer.add_scalar("Discriminator Loss", total_d_loss / args.n_critic, i + 1)
        writer.add_scalar("Generator Loss", g_loss, i + 1)
        if (i + 1) % args.interval == 0:
            fake_images: Tensor = G(fixed_noise)
            img_grid = make_grid(denormalize(fake_images), nrow=8, padding=2)
            writer.add_image("Fake Images", img_grid, (i + 1) // args.interval)
            save_image(
                img_grid,
                os.path.join(
                    args.sample_dir,
                    args.dataset,
                    f"fake_images_{(i + 1) // args.interval}.png",
                ),
            )
    # Save the model checkpoints.
    torch.save(G.state_dict(), os.path.join(args.ckpt_dir, args.dataset, "G.ckpt"))
    torch.save(D.state_dict(), os.path.join(args.ckpt_dir, args.dataset, "D.ckpt"))


def main() -> None:
    args = get_args()
    setup(args)
    C = 1 if args.dataset in ("MNIST", "FashionMNIST") else 3
    data_loader = get_data_loader(args)
    # Generator and Discrminator.
    G = Generator(num_channels=C, latent_dim=args.latent_dim).to(device)
    D = Discriminator(num_channels=C).to(device)
    train(args, G, D, data_loader)


if __name__ == "__main__":
    main()
