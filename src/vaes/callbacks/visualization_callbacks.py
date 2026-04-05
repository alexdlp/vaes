from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from typing import Any, List, Optional, Tuple
from sklearn.manifold import TSNE
from vaes.logger import logger
from vaes.utils.visualization_utils import (
    plot_density_meshgrid,
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


class FlowDensityVizCallback(VizBaseCallback):
    """
    Callback that visualizes target and learned flow densities.

    - Exact target density: generated once in on_fit_start and saved immediately.
    - Flow density: regenerated every plot_every_n_epochs epochs via the standard
      collect_samples / create_figures cycle.

    Args:
        plot_every_n_epochs: Frequency of flow density visualization.
        lims:                Plot boundaries [[x_min, x_max], [y_min, y_max]].
        nb_points:           Grid resolution used for both plots.
        cmap:                Matplotlib colormap.
        prefix:              Artifact filename prefix.
    """

    def __init__(
        self,
        plot_every_n_epochs: int = 10,
        lims: List[List[float]] = ((-4, 4), (-4, 4)),
        nb_points: int = 500,
        cmap: str = "coolwarm",
        prefix: str = "flow_density",
    ) -> None:
        super().__init__(plot_every_n_epochs=plot_every_n_epochs, prefix=prefix)
        self.lims = np.array(lims)
        self.nb_points = nb_points
        self.cmap = cmap
        self._flow_title: str = "Flow density"

    def generate_meshgrid(self, lims=None) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        lims = self.lims if lims is None else np.array(lims)
        xs = np.linspace(lims[0][0], lims[0][1], self.nb_points)
        ys = np.linspace(lims[1][0], lims[1][1], self.nb_points)
        xx, yy = np.meshgrid(xs, ys)
        z_grid = torch.tensor(
            np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32
        )
        return z_grid, xx, yy

    def on_fit_start(self, pipeline) -> None:
        super().on_fit_start(pipeline)

        energy_name = pipeline.energy_fn.__name__          # e.g. "U1"
        energy_idx  = energy_name[1:]                      # e.g. "1"
        K = len(pipeline.model.flows)

        exact_title = rf"$\exp(-U_{{{energy_idx}}}(z))$"
        self._flow_title = rf"$q_{{K={K}}}(z_K)$  —  {energy_name}"

        z_grid, xx, yy = self.generate_meshgrid()

        with torch.no_grad():
            density = (
                torch.exp(pipeline.criterion.log_prob_fn(z_grid.to(pipeline.device)))
                .reshape(self.nb_points, self.nb_points)
                .cpu()
                .numpy()
            )

        fig = plot_density_meshgrid(
            x=xx,
            y=yy,
            density=density,
            lims=self.lims,
            cmap=self.cmap,
            title=exact_title,
        )

        target_dir = self.save_dir if self.save_dir else pipeline._get_run_artifacst_dir()
        pdf_path = self._save_pdf([fig], Path(target_dir) / "target_density.pdf")
        logger.info(f"[{self.__class__.__name__}] Saved target density: {pdf_path}")

    def collect_samples(self, pipeline) -> List[Any]:
        was_training = pipeline.model.training
        pipeline.model.eval()

        # z0 se muestrea en un rango amplio para cubrir todo el soporte de la
        # base distribution; lims solo controla el recorte del display.
        z0, _, _ = self.generate_meshgrid(lims=[[-15, 15], [-15, 15]])

        with torch.no_grad():
            zK, log_jacobian = pipeline.model(z0.to(pipeline.device))
            log_q0 = pipeline.criterion.base_dist.log_prob(z0).sum(-1)
            log_qK = log_q0 - log_jacobian.cpu()
            qK = torch.exp(log_qK).reshape(self.nb_points, self.nb_points).numpy()

        zK_np = zK.cpu().numpy()

        if was_training:
            pipeline.model.train()

        return [{
            "x": zK_np[:, 0].reshape(self.nb_points, self.nb_points),
            "y": zK_np[:, 1].reshape(self.nb_points, self.nb_points),
            "density": qK,
        }]

    def create_figures(self, samples: List[Any]) -> List[Figure]:
        if not samples:
            return []

        data = samples[0]
        return [plot_density_meshgrid(
            x=data["x"],
            y=data["y"],
            density=data["density"],
            lims=self.lims,
            cmap=self.cmap,
            title=self._flow_title,
        )]
