import torch
import torch.nn as nn


class KLDivergence(nn.Module):
    """
    Computes the exact KL divergence between two distributions using PyTorch's
    analytical formulas.

    Only applicable when both distributions are torch.distributions objects and
    a closed-form KL formula is registered for that pair. Supported pairs can be
    checked via torch.distributions.kl._KL_REGISTRY.

    Note: Not suitable for normalizing flows, where q_K has no closed form.
    """

    def forward(self, p: torch.distributions.Distribution,
                      q: torch.distributions.Distribution) -> torch.Tensor:
        """
        Args:
            p: Source distribution (torch.distributions object).
            q: Target distribution (torch.distributions object).

        Returns:
            KL divergence D_KL(p || q) computed analytically.

        Raises:
            NotImplementedError: If no analytical formula is registered for the
            given distribution pair.
        """
        return torch.distributions.kl_divergence(p, q)
    
class KLDivergenceMC(nn.Module):
    """General Monte Carlo KL estimate for any distribution from which you can sample."""

    def forward(self, log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log_q: Log probabilities under the approximate distribution q, shape (N,).
            log_p: Log probabilities under the target distribution p, shape (N,).

        Returns:
            Scalar Monte Carlo estimate of KL(q || p).
        """
        return (log_q - log_p).mean()


class KLDivergenceFlowMC(KLDivergenceMC):
    """
    Estimates the KL divergence between a normalizing flow posterior q_K and a
    target distribution p using Monte Carlo estimation:

        KL(q_K || p) = E_{z ~ q_K} [log q_K(z) - log p(z)]
                     ≈ mean(log_qK - log_p)    # averaged over batch samples

    Applicable to any distribution from which you can sample, including those
    without a closed form such as normalizing flow posteriors.

    Args:
        log_prob_fn: Computes log p(z) for the target distribution. Can be any
                     callable — e.g. a negated energy function lambda z: -U(z),
                     or a torch.distributions log_prob method.
        base_dist: Base distribution q_0 from which z_0 is sampled. Must have a
                   log_prob method. Defaults to a standard Normal N(0, 1).
    """

    def __init__(self, log_prob_fn: callable,
                       base_dist: torch.distributions.Distribution = None):
        super().__init__()
        self.log_prob_fn = log_prob_fn
        self.base_dist = base_dist or torch.distributions.Normal(0, 1)


    def forward(self, z0: torch.Tensor, zK: torch.Tensor,
                      log_jacobian: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z0:           Samples from the base distribution q_0, shape (N, D).
            zK:           Samples after applying the full flow, shape (N, D).
            log_jacobian: Accumulated sum of log|det J_k| across all flow steps,
                          shape (N,). Encodes the volume change introduced by the flow.

        Returns:
            Scalar Monte Carlo estimate of KL(q_K || p).
        """
        # log q_0(z_0) — log prob of samples under the base distribution
        log_q0 = self.base_dist.log_prob(z0).sum(-1)

        # log q_K(z_K) = log q_0(z_0) - sum_k log|det J_k|  (change of variables, eq. 7)
        log_qK = log_q0 - log_jacobian

        # log p(z_K) — log prob of transformed samples under the target
        log_p = self.log_prob_fn(zK)

        # Monte Carlo estimate of E_{z ~ q_K} [log q_K(z) - log p(z)]
        return super().forward(log_qK, log_p)


