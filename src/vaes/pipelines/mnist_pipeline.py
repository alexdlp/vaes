"""Training pipeline for autoencoder / VAE models on MNIST."""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vaes.pipelines import register_pipeline
from vaes.pipelines.base_pipeline import BasePipeline
from vaes.callbacks import Callback, ModelCheckpoint, EarlyStopping, MNISTVizCallback
from vaes.dataloaders import create_mnist_dataloaders
from vaes.models import MODEL_REGISTRY


@register_pipeline("mnist")
class MNISTPipeline(BasePipeline):
    """Pipeline for training AE / VAE models on the MNIST dataset.

    Supports both deterministic autoencoders (output dict has ``z`` and
    ``recon``) and variational autoencoders (output dict additionally
    contains ``mu`` and ``logvar``).  The loss function is chosen via
    Hydra config — use ``loss: vae`` for VAE training and ``loss: mse``
    for vanilla autoencoders.
    """

    # ------------------------------------------------------------------ data
    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        return create_mnist_dataloaders(**self.cfg.data)

    # ----------------------------------------------------------------- model
    def build_model(self) -> nn.Module:
        arch = self.cfg.model.arch
        if arch not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown architecture '{arch}'. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )
        return MODEL_REGISTRY[arch](**self.cfg.model.params)

    # -------------------------------------------------------------- training
    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        images, _labels = batch
        out = self.model(images)

        if "mu" in out:
            # VAE path: criterion expects (x, x_hat, mu, logvar)
            loss, components = self.criterion(images, out["recon"], out["mu"], out["logvar"])
            return {"loss": loss, **components}
        else:
            # Vanilla AE path: standard reconstruction loss
            loss = self.criterion(out["recon"], images)
            return {"loss": loss}

    # ------------------------------------------------------------ validation
    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        images, _labels = batch
        out = self.model(images)

        if "mu" in out:
            loss, components = self.criterion(images, out["recon"], out["mu"], out["logvar"])
            return {"loss": loss, **components}
        else:
            loss = self.criterion(out["recon"], images)
            return {"loss": loss}

    # ------------------------------------------------------------ callbacks
    def init_callbacks(self) -> list[Callback]:
        callbacks: list[Callback] = []

        if hasattr(self.cfg.callbacks, "model_checkpoint"):
            callbacks.append(ModelCheckpoint(**self.cfg.callbacks.model_checkpoint))

        if hasattr(self.cfg.callbacks, "early_stopping"):
            callbacks.append(EarlyStopping(**self.cfg.callbacks.early_stopping))

        # Visualization callback (always active)
        callbacks.append(MNISTVizCallback(n_images=8, plot_every_n_epochs=5))

        return callbacks
