from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataLoader(DataLoader):
    """
    DataLoader for MNIST dataset.
    Downloads MNIST, resizes to 32x32, and wraps in a standard DataLoader.
    """

    def __init__(self, data_dir="./data/mnist", train=True,
                 batch_size=64, shuffle=True, num_workers=4):

        tensor_transforms = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

        dataset = MNIST(
            root=data_dir,
            train=train,
            transform=tensor_transforms,
            download=True,
        )

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
