from typing import Any, Dict, Tuple
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from vaes.pipelines import register_pipeline
from vaes.pipelines.base_pipeline import BasePipeline, Callback
from vaes.dataloaders import MNISTDataLoader
from vaes.models import LinearVAE
from vaes.callbacks import ModelCheckpoint, EarlyStopping


@register_pipeline("linear_vae")
class LinearVAEPipeline(BasePipeline):

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        train_dataloader = MNISTDataLoader(**self.cfg.data.train)
        val_dataloader = MNISTDataLoader(**self.cfg.data.val)
        return train_dataloader, val_dataloader

    def build_model(self) -> nn.Module:
        return LinearVAE(**self.cfg.model.params)

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
        model_checkpoint = ModelCheckpoint(**self.cfg.callbacks.model_checkpoint)
        early_stopping = EarlyStopping(**self.cfg.callbacks.early_stopping)
        return [model_checkpoint, early_stopping]
