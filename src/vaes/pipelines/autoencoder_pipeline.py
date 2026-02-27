from typing import Any, Dict, Tuple
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from itp_fabadII.pipelines import register_pipeline
from itp_fabadII.pipelines.base_pipeline import BasePipeline, Callback
from itp_fabadII.dataloaders import CubeDataLoader
from itp_fabadII.models import Autoencoder
from itp_fabadII.callbacks import ModelCheckpoint, EarlyStopping, CubeAEVizCallback

@register_pipeline("autoencoder")
class CubeTrainerPipeline(BasePipeline):
    """
    Minimal working pipeline for CubeH5Dataset training using Fabric.
    Handles model setup, training, validation, and MLflow logging.
    """

    # ---- data ----
    def load_data(self) -> Tuple[DataLoader, DataLoader]:

        train_dataloader = CubeDataLoader(**self.cfg.data.train)
        val_dataloader = CubeDataLoader(**self.cfg.data.val)

        return train_dataloader, val_dataloader

    # ---- model ----
    def build_model(self) -> nn.Module:
        """
        Simple example model. Adjust to match your feature and output dimensions.
        """
        model = Autoencoder(**self.cfg.model.params)
        return model


    # ---- training ----
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        One training step.
        """

        x = batch["meltpool_seq"]

        preds = self.model(x)
        loss = self.criterion(preds, x)

        return {"loss": loss}

    # ---- validation ----
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        
        x = batch["meltpool_seq"]

        preds = self.model(x)
        loss = self.criterion(preds, x)

        return {"loss": loss}
    
    def init_callbacks(self) -> list[Callback]: 

        model_checkpoint = ModelCheckpoint(**self.cfg.callbacks.model_checkpoint)
        earlyStopping = EarlyStopping(**self.cfg.callbacks.early_stopping)
        aevizcallback = CubeAEVizCallback(**self.cfg.callbacks.ae_viz_callback)
        
        return [model_checkpoint, earlyStopping, aevizcallback]
    


    
