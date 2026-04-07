from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.manifold import TSNE

from vaes.logger import logger
from vaes.utils.viz_vae import (
    plot_generated_samples,
    plot_latent_interpolation_grid,
    plot_latent_scatter,
    plot_reconstruction_comparison,
)
from .viz_base_callback import VizBaseCallback


def _extract_latent_and_reconstruction(model_output: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return `(z, x_hat)` from AE/VAE model outputs."""
    if not isinstance(model_output, tuple):
        raise ValueError("Model output is not a tuple. Expected (z, x_hat, ...).")
    if len(model_output) < 2:
        raise ValueError("Model output tuple has fewer than 2 elements.")
    return model_output[0], model_output[1]


def _to_image_batch(decoded: torch.Tensor, image_size: int = 32) -> torch.Tensor:
    """Normalize decoder output to `[N, 1, H, W]` when possible."""
    if decoded.ndim == 4:
        return decoded

    if decoded.ndim == 2 and decoded.shape[-1] == image_size * image_size:
        return decoded.reshape(decoded.shape[0], 1, image_size, image_size)

    raise ValueError(
        f"Unsupported decoded tensor shape {tuple(decoded.shape)} for visualization."
    )


class ReconstructionVizCallback(VizBaseCallback):
    """
    Callback that logs original-vs-reconstruction panels as PDF artifacts.
    """

    def __init__(
        self,
        plot_every_n_epochs: int = 10,
        n_samples: int = 8,
        prefix: str = "reconstructions",
    ) -> None:
        super().__init__(plot_every_n_epochs=plot_every_n_epochs, prefix=prefix)
        self.n_samples = n_samples

    def collect_samples(self, pipeline) -> List[Any]:
        """Collect a single validation batch for reconstruction visualization."""
        was_training = pipeline.model.training
        pipeline.model.eval()

        samples: List[Any] = []
        with torch.no_grad():
            for batch in pipeline.val_loader:
                images, labels = batch
                images = images[: self.n_samples].to(pipeline.device)

                output = pipeline.model(images)
                _, recon = _extract_latent_and_reconstruction(output)

                samples.append(
                    {
                        "images": images.detach().cpu(),
                        "recon": recon.detach().cpu(),
                    }
                )
                break

        if was_training:
            pipeline.model.train()
        return samples

    def create_figures(self, samples: List[Any]) -> List[Figure]:
        """Create reconstruction figures using reusable plotting helpers."""
        if not samples:
            return []

        entry = samples[0]
        images = entry["images"]
        recon = entry["recon"]
        fig = plot_reconstruction_comparison(
            images=images.numpy(),
            reconstructions=recon.numpy(),
            max_samples=self.n_samples,
        )
        return [fig]


class LatentSpaceVizCallback(VizBaseCallback):
    """
    Callback that logs latent-space scatter plots.

    It uses:
    - raw latent coordinates when dimension is 2
    - t-SNE for higher dimensions (same approach as notebook)
    """

    def __init__(
        self,
        plot_every_n_epochs: int = 10,
        max_points: int = 2048,
        prefix: str = "latent_space",
    ) -> None:
        super().__init__(plot_every_n_epochs=plot_every_n_epochs, prefix=prefix)
        self.max_points = max_points

    def collect_samples(self, pipeline) -> List[Any]:
        """Collect latent vectors and labels from validation data."""
        was_training = pipeline.model.training
        pipeline.model.eval()

        latents: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []

        with torch.no_grad():
            points = 0
            for batch in pipeline.val_loader:
                images, y = batch
                images = images.to(pipeline.device)

                output = pipeline.model(images)
                z, _ = _extract_latent_and_reconstruction(output)
                z = z.reshape(z.shape[0], -1).detach().cpu()

                latents.append(z)
                labels.append(y.detach().cpu())
                points += z.shape[0]

                if points >= self.max_points:
                    break

        if was_training:
            pipeline.model.train()

        if not latents:
            return []

        z_all = torch.cat(latents, dim=0)[: self.max_points].numpy()
        y_all = torch.cat(labels, dim=0)[: self.max_points].numpy()
        return [{"latents": z_all, "labels": y_all}]

    def create_figures(self, samples: List[Any]) -> List[Figure]:
        """Create latent-space scatter figures using reusable plotting helpers."""
        if not samples:
            return []

        data = samples[0]
        z = data["latents"]
        y = data["labels"]

        if z.shape[1] < 2:
            logger.warning("LatentSpaceViz: latent dim < 2. Skipping.")
            return []

        if z.shape[1] == 2:
            coords = z
            title_suffix = "2D Latents"
        elif z.shape[0] > 3:
            perplexity = min(30, z.shape[0] - 1)
            coords = TSNE(
                n_components=2,
                n_jobs=-1,
                random_state=42,
                perplexity=perplexity,
            ).fit_transform(z)
            title_suffix = "t-SNE"
        else:
            logger.warning(
                "LatentSpaceViz: not enough samples to run t-SNE (need >3). Skipping."
            )
            return []

        fig = plot_latent_scatter(coords_2d=coords, labels=y, title=f"Latent Space ({title_suffix})")
        return [fig]


class LatentInterpolationVizCallback(VizBaseCallback):
    """
    Callback that logs latent interpolation grids for 2D latent models.
    """

    def __init__(
        self,
        plot_every_n_epochs: int = 10,
        grid_size: int = 12,
        x_range: Tuple[float, float] = (-2.0, 2.0),
        y_range: Tuple[float, float] = (-2.0, 2.0),
        prefix: str = "latent_interpolation",
    ) -> None:
        super().__init__(plot_every_n_epochs=plot_every_n_epochs, prefix=prefix)
        self.grid_size = grid_size
        self.x_range = x_range
        self.y_range = y_range

    def _infer_latent_shape(self, pipeline) -> Optional[Tuple[int, ...]]:
        """Infer latent shape from one validation forward pass."""
        with torch.no_grad():
            for batch in pipeline.val_loader:
                images, _ = batch
                images = images[:1].to(pipeline.device)
                output = pipeline.model(images)
                z, _ = _extract_latent_and_reconstruction(output)
                return tuple(z.shape[1:])
        return None

    def collect_samples(self, pipeline) -> List[Any]:
        """Decode a regular 2D latent grid into image samples."""
        latent_shape = self._infer_latent_shape(pipeline)
        if latent_shape is None:
            return []

        if int(np.prod(latent_shape)) != 2:
            logger.warning(
                f"LatentInterpolationViz: expected latent size 2, got shape {latent_shape}. Skipping."
            )
            return []

        xs = np.linspace(self.x_range[0], self.x_range[1], self.grid_size)
        ys = np.linspace(self.y_range[0], self.y_range[1], self.grid_size)

        latent_points = []
        for y in ys:
            for x in xs:
                latent_points.append([x, y])

        z = torch.tensor(latent_points, dtype=torch.float32, device=pipeline.device)

        was_training = pipeline.model.training
        pipeline.model.eval()
        with torch.no_grad():
            decoded = pipeline.model.decoder(z)
        if was_training:
            pipeline.model.train()

        images = _to_image_batch(decoded).detach().cpu()
        return [{"images": images, "grid_size": self.grid_size}]

    def create_figures(self, samples: List[Any]) -> List[Figure]:
        """Create interpolation-grid figures using reusable plotting helpers."""
        if not samples:
            return []

        entry = samples[0]
        images = entry["images"]
        grid_size = entry["grid_size"]
        fig = plot_latent_interpolation_grid(images=images.numpy(), grid_size=grid_size)
        return [fig]


class RandomGenerationVizCallback(VizBaseCallback):
    """
    Callback that logs random samples produced from Gaussian latent noise.
    """

    def __init__(
        self,
        plot_every_n_epochs: int = 10,
        n_samples: int = 16,
        prefix: str = "generated_samples",
    ) -> None:
        super().__init__(plot_every_n_epochs=plot_every_n_epochs, prefix=prefix)
        self.n_samples = n_samples

    def _infer_latent_shape(self, pipeline) -> Optional[Tuple[int, ...]]:
        """Infer latent shape from one validation forward pass."""
        with torch.no_grad():
            for batch in pipeline.val_loader:
                images, _ = batch
                images = images[:1].to(pipeline.device)
                output = pipeline.model(images)
                z, _ = _extract_latent_and_reconstruction(output)
                return tuple(z.shape[1:])
        return None

    def collect_samples(self, pipeline) -> List[Any]:
        """Sample Gaussian latent noise and decode generated images."""
        latent_shape = self._infer_latent_shape(pipeline)
        if latent_shape is None:
            return []

        noise = torch.randn((self.n_samples, *latent_shape), device=pipeline.device)

        was_training = pipeline.model.training
        pipeline.model.eval()
        with torch.no_grad():
            decoded = pipeline.model.decoder(noise)
        if was_training:
            pipeline.model.train()

        images = _to_image_batch(decoded).detach().cpu()
        return [{"images": images}]

    def create_figures(self, samples: List[Any]) -> List[Figure]:
        """Create random-generation figures using reusable plotting helpers."""
        if not samples:
            return []

        images = samples[0]["images"]
        fig = plot_generated_samples(images=images.numpy(), max_samples=self.n_samples)
        return [fig]
