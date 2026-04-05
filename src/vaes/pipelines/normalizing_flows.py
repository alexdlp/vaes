from typing import Any, Dict, Tuple
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from hydra.utils import instantiate, get_method
from vaes.pipelines import register_pipeline
from vaes.pipelines.base_pipeline import BasePipeline, Callback
from vaes.dataloaders import DistributionDataLoader
from vaes.models import NormalizingFlow
from vaes.utils.config_utils import ConfigNamespace
from vaes.losses.kl_loss import KLDivergenceFlowMC
from vaes.callbacks import (
    EarlyStopping,
    FlowDensityVizCallback,
    ModelCheckpoint,
)


@register_pipeline("planar_flows")
class NormalizingFlowsPipeline(BasePipeline):

    def load_data(self) -> Tuple[DataLoader, DataLoader]:

        train_cfg = dict(self.cfg.data.train)
        val_cfg = dict(self.cfg.data.val)

        # train_cfg.update(
        #     {"distribution": instantiate(self.cfg.data.train.distribution)}
        #     )
        # val_cfg.update(
        #     {"distribution": instantiate(self.cfg.data.val.distribution)}
        #     )
        
        train_cfg.update(
            {"distribution": instantiate(ConfigNamespace.to_builtin(self.cfg.data.train.distribution))}
            )
        val_cfg.update(
            {"distribution": instantiate(ConfigNamespace.to_builtin(self.cfg.data.val.distribution))}
            )

        train_dataloader = DistributionDataLoader(**train_cfg)
        val_dataloader = DistributionDataLoader(**val_cfg)
        return train_dataloader, val_dataloader

    def build_model(self) -> nn.Module:
        return NormalizingFlow(**self.cfg.model.params)
    
    def build_loss(self):
        self.energy_fn = get_method(self.cfg.loss.energy_fn)
        base_dist = instantiate(ConfigNamespace.to_builtin(self.cfg.data.train.distribution))
        return KLDivergenceFlowMC(log_prob_fn = lambda z: -self.energy_fn(z),
                                  base_dist = base_dist)

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        z_0 = batch
        z_k, total_log_det = self.model(z_0)
        loss = self.criterion(z0 = z_0, zK = z_k, log_jacobian = total_log_det)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        z_0 = batch
        z_k, total_log_det = self.model(z_0)
        loss = self.criterion(z0 = z_0, zK = z_k, log_jacobian = total_log_det)
        return {"loss": loss}

    def init_callbacks(self) -> list[Callback]:
        model_checkpoint = ModelCheckpoint(**self.cfg.callbacks.model_checkpoint)
        early_stopping = EarlyStopping(**self.cfg.callbacks.early_stopping)
        flow_density_viz = FlowDensityVizCallback(**self.cfg.callbacks.flow_density_viz)

        return [
            model_checkpoint,
            early_stopping,
            flow_density_viz,
        ]
