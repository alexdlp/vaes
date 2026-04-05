import torch
import torch.nn as nn
import torch.nn.functional as F

class PlanarFlow(nn.Module):
    """
    Planar normalizing flow from Rezende & Mohamed (2015), eqs. (10)-(13).

    Transformation:
        f(z) = z + u_hat * h(w^T z + b)          # eq. (10)

    Log-det-Jacobian (via matrix determinant lemma):
        log |det df/dz| = log |1 + u_hat^T psi(z)| # eq. (12)
    where
        psi(z) = h'(w^T z + b) * w                # eq. (11)

    Invertibility constraint (Appendix A.1):
        w^T u_hat >= -1
    enforced via the reparametrization:
        u_hat = u + (m(w^T u) - w^T u) * w / ||w||^2
    where m(x) = -1 + log(1 + exp(x))
    """

    def __init__(self, dims: int):
        super().__init__()
        self.w = nn.Parameter(torch.rand(dims))
        self.u = nn.Parameter(torch.rand(dims))
        self.b = nn.Parameter(torch.rand(1))

    # ------------------------------------------------------------------
    # Appendix A.1 — reparametrize u to enforce w^T u_hat >= -1
    # ------------------------------------------------------------------
    def u_hat(self) -> torch.Tensor:
        """
        Returns u_hat such that w^T u_hat >= -1 (invertibility).

        u_hat = u + [m(w^T u) - (w^T u)] * w / ||w||^2
        where m(x) = -1 + softplus(x)
        """
        wu = self.w @ self.u                          # scalar: w^T u
        m_wu = -1 + F.softplus(wu)                   # m(x) = -1 + log(1 + e^x)
        return self.u + (m_wu - wu) * self.w / (self.w @ self.w)

    # ------------------------------------------------------------------
    # eq. (11): psi(z) = h'(w^T z + b) * w
    # ------------------------------------------------------------------
    def psi(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, D)
        returns psi: (B, D)
        """
        # w^T z + b: (B,)
        linear = z @ self.w + self.b                  # (B,)
        # tanh' = 1 - tanh^2
        h_prime = 1 - torch.tanh(linear) ** 2         # (B,)
        return h_prime.unsqueeze(1) * self.w           # (B, D)

    # ------------------------------------------------------------------
    # eq. (10) + eq. (12)
    # ------------------------------------------------------------------
    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the planar flow transformation.
        
        Args:
            z:       (B, D) — input samples
        Returns:
            z_next:  (B, D) — transformed samples, eq. (10)
            log_det: (B,)   — log |det df/dz| per sample, eq. (12)
        """
        u_hat = self.u_hat()                           # (D,)
        linear = z @ self.w + self.b                  # (B,)

        # eq. (10): f(z) = z + u_hat * h(w^T z + b)
        z_next = z + u_hat * torch.tanh(linear).unsqueeze(1)  # (B, D)

        # eq. (12): log |1 + u_hat^T psi(z)|
        # psi: (B, D) → u_hat^T psi: (B,) via einsum
        u_hat_psi = torch.einsum('d,bd->b', u_hat, self.psi(z))  # (B,)
        log_det = torch.log(torch.abs(1 + u_hat_psi) + 1e-8)     # (B,)

        return z_next, log_det
    
class NormalizingFlow(nn.Module):
    def __init__(self, dims: int, K: int):
        super().__init__()
        self.flows = nn.ModuleList([PlanarFlow(dims) for _ in range(K)])

    def forward(self, z0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns z_K and sum_k log|det df_k/dz_{k-1}|  # eq. (13)
        """
        z = z0
        total_log_det = torch.zeros(z.shape[0], device=z.device)
        for flow in self.flows:
            z, log_det = flow(z)
            total_log_det += log_det        # accumulate eq. (13) sum
        return z, total_log_det