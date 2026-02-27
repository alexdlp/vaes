"""Visualization callback for MNIST autoencoder / VAE training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, TYPE_CHECKING

import matplotlib.pyplot as plt
import mlflow
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from vaes.logger import logger
from .callbacks import Callback

if TYPE_CHECKING:
    from vaes.pipelines.base_pipeline import BasePipeline


class MNISTVizCallback(Callback):
    """Generate reconstruction and generation visualizations during training.

    At every ``plot_every_n_epochs`` epochs this callback:
    1. Takes a fixed batch from the validation set and plots original vs
       reconstructed images side by side.
    2. If the model is a VAE (output contains ``mu``), generates images
       from random Gaussian noise.

    The resulting figures are saved as a PDF in the MLflow artifact directory.

    Args:
        n_images: Number of images to include in the reconstruction grid.
        plot_every_n_epochs: Frequency of visualization (in epochs).
    """

    def __init__(self, n_images: int = 8, plot_every_n_epochs: int = 5) -> None:
        super().__init__()
        self.n_images = n_images
        self.plot_every_n_epochs = plot_every_n_epochs
        self._fixed_batch: tuple[torch.Tensor, torch.Tensor] | None = None

    def on_fit_start(self, pipeline: BasePipeline) -> None:
        """Cache a fixed batch of validation images for consistent comparison."""
        batch = next(iter(pipeline.val_loader))
        images, labels = batch
        self._fixed_batch = (images[: self.n_images], labels[: self.n_images])

    def on_epoch_end(self, pipeline: BasePipeline, epoch: int, logs: Dict[str, float]) -> None:
        epoch_1 = epoch + 1  # display as 1-indexed
        if epoch_1 != 1 and epoch_1 % self.plot_every_n_epochs != 0:
            return

        logger.info(f"[MNISTViz] Generating visualizations for epoch {epoch_1}...")

        figs = self._create_figures(pipeline)
        self._save_pdf(figs, pipeline, epoch_1)

        for fig in figs:
            plt.close(fig)

    # ------------------------------------------------------------------

    def _create_figures(self, pipeline: BasePipeline) -> list[Figure]:
        """Create reconstruction (and optionally generation) figures."""
        figs: list[Figure] = []

        images, _labels = self._fixed_batch
        images = images.to(pipeline.device)

        pipeline.model.eval()
        with torch.no_grad():
            out = pipeline.model(images)
        pipeline.model.train()

        recon = out["recon"].cpu()
        images_cpu = images.cpu()

        # --- Reconstruction grid ---
        fig, axes = plt.subplots(2, self.n_images, figsize=(2 * self.n_images, 4))
        fig.suptitle("Reconstruction: original (top) vs decoded (bottom)")
        for i in range(self.n_images):
            axes[0, i].imshow(images_cpu[i].squeeze(), cmap="gray")
            axes[0, i].axis("off")
            axes[1, i].imshow(recon[i].squeeze().clamp(0, 1), cmap="gray")
            axes[1, i].axis("off")
        fig.tight_layout()
        figs.append(fig)

        # --- Generation from random noise (VAE only) ---
        if "mu" in out and hasattr(pipeline.model, "decoder"):
            fig_gen = self._generate_samples(pipeline)
            if fig_gen is not None:
                figs.append(fig_gen)

        return figs

    def _generate_samples(self, pipeline: BasePipeline) -> Figure | None:
        """Generate images by sampling random latent vectors from N(0, I)."""
        model = pipeline.model
        model.eval()

        # Determine latent shape by looking at a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, *self._fixed_batch[0].shape[1:], device=pipeline.device)
            dummy_out = model(dummy)
            z_shape = dummy_out["z"].shape[1:]  # latent shape without batch

        # Sample from standard normal (flat latent space)
        z = torch.randn(self.n_images, *z_shape, device=pipeline.device)

        # Conv decoders expect 4D input (B, C, H', W'); reshape if needed
        if hasattr(model.decoder, "channels_bottleneck"):
            cb = model.decoder.channels_bottleneck
            spatial = int((z.shape[1] / cb) ** 0.5)
            z = z.reshape(self.n_images, cb, spatial, spatial)

        with torch.no_grad():
            generated = model.decoder(z)

        model.train()

        # Linear decoder returns flat vector; reshape to image
        generated = generated.cpu()
        if generated.dim() == 2:
            side = int(generated.shape[1] ** 0.5)
            generated = generated.reshape(-1, 1, side, side)

        fig, axes = plt.subplots(1, self.n_images, figsize=(2 * self.n_images, 2))
        fig.suptitle("Generated samples from N(0, I)")
        for i in range(self.n_images):
            axes[i].imshow(generated[i].squeeze().clamp(0, 1), cmap="gray")
            axes[i].axis("off")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------

    @staticmethod
    def _save_pdf(figs: list[Figure], pipeline: BasePipeline, epoch: int) -> None:
        """Save figures as a multi-page PDF in the MLflow artifact directory."""
        artifact_dir = Path(pipeline._get_run_artifacst_dir())
        pdf_path = artifact_dir / f"viz_epoch_{epoch}.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        with PdfPages(pdf_path) as pdf:
            for fig in figs:
                pdf.savefig(fig)

        logger.info(f"[MNISTViz] Saved visualization: {pdf_path}")
