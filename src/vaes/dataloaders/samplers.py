import torch

class DistributionDataset(torch.utils.data.Dataset):
    """
    A Dataset that samples from a given distribution on-the-fly.
    Each __getitem__ call generates a fresh sample, so data is never
    repeated across epochs.

    Args:
        distribution: Any torch.distributions object with a sample() method.
        n_samples:    Virtual dataset size — controls epoch length, not data diversity.
        dims:         Dimensionality of each sample.
    """

    def __init__(self, distribution: torch.distributions.Distribution, n_samples: int, dims: int):
        self.distribution = distribution
        self.n_samples = int(n_samples)
        self.dims = int(dims)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.distribution.sample((self.dims,))


class DistributionDataLoader(torch.utils.data.DataLoader):
    """
    A DataLoader that samples from a given distribution on-the-fly.
    Instantiates a DistributionDataset internally.

    Args:
        distribution: Any torch.distributions object with a sample() method.
        n_samples:    Virtual dataset size — controls epoch length, not data diversity.
        dims:          Dimensionality of each sample.
        batch_size:   Number of samples per batch.
    """

    def __init__(self, distribution: torch.distributions.Distribution,
                       n_samples: int,
                       dims: int, 
                       batch_size: int = 512,
                       **kwargs):
        kwargs.pop("shuffle", None)
        dataset = DistributionDataset(distribution, n_samples, dims)
        super().__init__(dataset, batch_size=batch_size, shuffle=False, **kwargs)