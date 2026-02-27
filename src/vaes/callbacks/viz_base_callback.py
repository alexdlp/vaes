from __future__ import annotations

from abc import ABC, abstractmethod
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import h5py
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from itp_fabadII.logger import logger

from itp_fabadII.utils.scalers import load_scalers
from .callbacks import Callback


if TYPE_CHECKING:
    from itp_fabadII.pipelines.base_pipeline import BasePipeline
from abc import ABC, abstractmethod


class VizBaseCallback(Callback, ABC):
    """
    Base class for visualization callbacks in a PyTorch training pipeline.

    Handles interval logic, figure saving, and garbage collection.
    Subclasses implement:
        - collect_samples(pipeline)
        - create_figures(samples)
    """

    def __init__(self, val_data_dir: str, scalers: Optional[dict] = None, plot_every_n_epochs: int = 10,
        save_dir: Optional[str] = None, prefix: str = "viz_epoch") -> None:
        super().__init__()
        self.val_data_dir = Path(val_data_dir)
        self.scalers_config = scalers
        self.plot_every_n_epochs = plot_every_n_epochs
        self.prefix = prefix

        # Save directory (local)
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Load scalers if provided
        self._load_scalers()
    
    # Scalers
    def _load_scalers(self) -> None:
        base_dir = self.val_data_dir.parent
        if self.scalers_config:
            scalers = load_scalers(base_dir, file_mapping=self.scalers_config)
            for attr_name, scaler_obj in scalers.items():
                setattr(self, attr_name, scaler_obj)

    # Saving
    def _get_mlflow_path(self, filename: str) -> str:
        """
        Gets the local path for saving artifacts in MLflow.
        """
        artifact_uri = Path(mlflow.active_run().info.artifact_uri)
        artifact_uri_local = artifact_uri.as_posix().replace('file:', '')  # Remove 'file:' if needed
        artifact_path = Path(artifact_uri_local) / filename
        return artifact_path.as_posix()

 
    def _save_pdf(self, figs: List[Figure], pdf_path: Path) -> Path:
        """
        Saves a list of figures into a single PDF at the specified absolute path.
        This method ignores `self.save_dir` and uses the provided path directly.

        Args:
            figs: A list of Matplotlib Figure objects.
            pdf_path: The full, absolute path where the PDF will be saved.

        Returns:
            The path where the PDF was saved.
        """
        # Ensure the parent directory exists before writing the file
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        with PdfPages(pdf_path) as pdf:
            for fig in figs:
                pdf.savefig(fig)
                plt.close(fig)

        return pdf_path

    def _save_pdf_from_samples(self, samples: List[Any], pdf_path: Path) -> Path:
        """
        Saves figures incrementally by sample to keep peak open-figure count low.
        """
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        with PdfPages(pdf_path) as pdf:
            for sample in samples:
                figs = self.create_figures([sample])
                for fig in figs:
                    pdf.savefig(fig)
                    plt.close(fig)

        return pdf_path
    
    # -------------------------------------------------------------------------
    # Hooks
    # -------------------------------------------------------------------------
    def on_fit_start(self, pipeline: "BasePipeline") -> None:
        """
        Called before training begins.
        """
        if pipeline.model is None:
            raise ValueError("Pipeline model is None. Did you forget to assign it?")
        logger.info("Model attached to visualization callback.")


    def on_epoch_end(self, pipeline: "BasePipeline", epoch: int, logs: Dict[str, float]) -> None:
        """
        Standard visualization workflow:
            epoch_filter -> collect_samples -> create_figures -> save_pdf
        """

        epoch = epoch + 1

        if epoch == 1 or epoch % self.plot_every_n_epochs == 0:
            
        
            logger.info(f"Generating validation plots for epoch {epoch}...")

            samples = self.collect_samples(pipeline=pipeline)

            target_dir = self.save_dir if self.save_dir else pipeline._get_run_artifacst_dir()

            filename = Path(target_dir) / f"{self.prefix}_{epoch}.pdf"
            pdf_path = self._save_pdf_from_samples(samples, filename)
            logger.info(f"Saved visualization: {pdf_path}")

            plt.close('all')
            gc.collect()


    def on_fit_end(self, pipeline: "BasePipeline") -> None:
        """
        Called when training ends.
        """
        total_epochs = pipeline.last_epoch
        already_plotted_last_epoch = (
            total_epochs == 1 or total_epochs % self.plot_every_n_epochs == 0
        )

        if already_plotted_last_epoch:
            logger.info(
                f"Training completed after {total_epochs} epochs. "
                "Skipping final validation plots (already generated on last epoch)."
            )
            return

        logger.info(f"Training completed after {total_epochs} epochs. Generating final validation plots...")
        self.on_epoch_end(pipeline, total_epochs - 1, logs={})

    # -------------------------------------------------------------------------
    # Subclass API
    # -------------------------------------------------------------------------
    @abstractmethod
    def collect_samples(self, pipeline: "BasePipeline") -> List[Any]:
        """
        Subclass must return a list of samples. A sample can be a dict, tuple,
        or custom object containing everything needed for visualization.
        """
        pass

    @abstractmethod
    def create_figures(self, samples: List[Any]) -> List[Figure]:
        """
        Subclass must return a list of matplotlib Figure objects.
        """
        pass
       
