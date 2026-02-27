from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def create_mnist_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 32,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders for the MNIST dataset.

    Images are resized to ``(image_size, image_size)`` and converted to
    tensors with pixel values in [0, 1].

    Args:
        data_dir: Root directory where the MNIST data is stored (or will be
            downloaded to).
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses for data loading.
        image_size: Spatial size to resize images to. Default is 32 to match
            the architecture used in the notebook.

    Returns:
        A tuple ``(train_loader, val_loader)``.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = MNIST(data_dir, train=True, download=True, transform=transform)
    val_dataset = MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
