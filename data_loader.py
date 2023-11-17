import torch
from torchvision import datasets
from torchvision import transforms


def get_loader(config, train_mode):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""

    svhn_transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    mnist_transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    svhn = datasets.SVHN(root=config.svhn_path, download=False, transform=svhn_transform)
    mnist = datasets.MNIST(root=config.mnist_path, download=False, transform=mnist_transform)

    svhn_loader = torch.utils.data.DataLoader(dataset=svhn,
                                              batch_size=config.batch_size,
                                              shuffle=train_mode,
                                              num_workers=config.num_workers)

    mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=config.batch_size,
                                               shuffle=train_mode,
                                               num_workers=config.num_workers)
    return svhn_loader, mnist_loader