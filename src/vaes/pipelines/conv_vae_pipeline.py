from typing import Any, Dict, Tuple
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from vaes.pipelines import register_pipeline
from vaes.pipelines.base_pipeline import BasePipeline, Callback
from vaes.dataloaders import MNISTDataLoader
from vaes.models import ConvVAE
from vaes.callbacks import (
    EarlyStopping,
    LatentInterpolationVizCallback,
    LatentSpaceVizCallback,
    ModelCheckpoint,
    RandomGenerationVizCallback,
    ReconstructionVizCallback,
)


@register_pipeline("conv_vae")
class ConvVAEPipeline(BasePipeline):

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        train_dataloader = MNISTDataLoader(**self.cfg.data.train)
        val_dataloader = MNISTDataLoader(**self.cfg.data.val)
        return train_dataloader, val_dataloader

    def build_model(self) -> nn.Module:
        return ConvVAE(**self.cfg.model.params)

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        images, labels = batch
        z, x_hat, mu, logvar = self.model(images)
        loss = self.criterion(images, x_hat, mu, logvar)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        images, labels = batch
        z, x_hat, mu, logvar = self.model(images)
        loss = self.criterion(images, x_hat, mu, logvar)
        return {"loss": loss}

    def init_callbacks(self) -> list[Callback]:
        """Initialize training callbacks for convolutional VAE training."""
        model_checkpoint = ModelCheckpoint(**self.cfg.callbacks.model_checkpoint)
        early_stopping = EarlyStopping(**self.cfg.callbacks.early_stopping)
        reconstruction_viz = ReconstructionVizCallback(**self.cfg.callbacks.reconstruction_viz)
        latent_space_viz = LatentSpaceVizCallback(**self.cfg.callbacks.latent_space_viz)
        latent_interpolation_viz = LatentInterpolationVizCallback(
            **self.cfg.callbacks.latent_interpolation_viz
        )
        random_generation_viz = RandomGenerationVizCallback(**self.cfg.callbacks.random_generation_viz)

        return [
            model_checkpoint,
            early_stopping,
            reconstruction_viz,
            latent_space_viz,
            latent_interpolation_viz,
            random_generation_viz,
        ]
