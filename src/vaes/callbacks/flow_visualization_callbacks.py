from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from typing import Any, List, Tuple

from vaes.logger import logger
from vaes.utils.viz_flow import plot_exact_density, plot_flow_density
from .viz_base_callback import VizBaseCallback


class FlowDensityVizCallback(VizBaseCallback):
    """
    Callback that visualizes target and learned flow densities.

    - Exact target density: generated once in on_fit_start and saved immediately.
    - Flow density: regenerated every plot_every_n_epochs epochs via the standard
      collect_samples / create_figures cycle.

    Args:
        plot_every_n_epochs: Frequency of flow density visualization.
        lims:                Plot boundaries [[x_min, x_max], [y_min, y_max]].
        nb_points_target:    Grid resolution for the target density plot.
        nb_points_flow:      Grid resolution for the flow density plot.
        cmap:                Matplotlib colormap.
        prefix:              Artifact filename prefix.
    """

    def __init__(
        self,
        plot_every_n_epochs: int = 10,
        lims: List[List[float]] = ((-4, 4), (-4, 4)),
        nb_points_target: int = 100,
        nb_points_flow: int = 1000,
        cmap: str = "coolwarm",
        prefix: str = "flow_density",
    ) -> None:
        super().__init__(plot_every_n_epochs=plot_every_n_epochs, prefix=prefix)
        self.lims = np.array(lims)
        self.nb_points_target = nb_points_target
        self.nb_points_flow = nb_points_flow
        self.cmap = cmap
        self._flow_title: str = "Flow density"

    def generate_meshgrid(self, lims=None, nb_points=None) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        lims = self.lims if lims is None else np.array(lims)
        n = nb_points or self.nb_points_flow
        xs = np.linspace(lims[0][0], lims[0][1], n)
        ys = np.linspace(lims[1][0], lims[1][1], n)
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

        z_grid, xx, yy = self.generate_meshgrid(nb_points=self.nb_points_target)

        with torch.no_grad():
            density = (
                torch.exp(pipeline.criterion.log_prob_fn(z_grid.to(pipeline.device)))
                .reshape(self.nb_points_target, self.nb_points_target)
                .cpu()
                .numpy()
            )

        fig = plot_exact_density(
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

        n = self.nb_points_flow
        z0, _, _ = self.generate_meshgrid(lims=[[-15, 15], [-15, 15]], nb_points=n)

        with torch.no_grad():
            zK, log_jacobian = pipeline.model(z0.to(pipeline.device))
            log_q0 = pipeline.criterion.base_dist.log_prob(z0).sum(-1)
            log_qK = log_q0 - log_jacobian.cpu()
            qK = torch.exp(log_qK).reshape(n, n).numpy()

        zK_np = zK.cpu().numpy()

        if was_training:
            pipeline.model.train()

        return [{
            "x": zK_np[:, 0].reshape(n, n),
            "y": zK_np[:, 1].reshape(n, n),
            "density": qK,
        }]

    def create_figures(self, samples: List[Any]) -> List[Figure]:
        if not samples:
            return []

        data = samples[0]
        return [plot_flow_density(
            x=data["x"],
            y=data["y"],
            density=data["density"],
            lims=self.lims,
            cmap=self.cmap,
            title=self._flow_title,
        )]
