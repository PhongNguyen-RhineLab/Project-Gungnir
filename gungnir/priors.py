"""Learned priors over the latent space.

ScoreNet + annealed Langevin is the GEN-CIM Phase 3b/4 component. The RealNVP
flow is the cheap competitor: same training data, one forward pass to sample.
Neither costs decoder calls at sampling time; only the pool that trains them does.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .device import cpu_gen, get_device, randn


# --------------------------------------------------------------------- score model

class ScoreNet(nn.Module):
    """s_theta(z, sigma) ~= grad_z log p_sigma(z). Sigma-conditioned MLP."""

    def __init__(self, dz: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dz + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dz),
        )

    def forward(self, z, sigma):
        s = torch.log(sigma).reshape(-1, 1)
        return self.net(torch.cat([z, s], dim=-1)) / sigma.reshape(-1, 1)


def train_scorenet(latents: np.ndarray, quality: np.ndarray, beta_s: float = 2.0,
                   sigma_min: float = 0.05, sigma_max: float = 1.0, epochs: int = 300,
                   batch: int = 256, lr: float = 1e-3, hidden: int = 512, seed: int = 0,
                   device: Optional[torch.device] = None) -> ScoreNet:
    """Denoising score matching, GEN-CIM eq. (9):
        L = E[ sigma^2 * || s(z + sigma*eps, sigma) + eps/sigma ||^2 ]
    Samples are drawn with weights ~ exp(beta_s * standardised quality), so the
    model spends capacity on the good part of the pool. `quality` is higher-better.
    """
    torch.manual_seed(seed)
    dev = device or get_device()
    Z = torch.as_tensor(latents, dtype=torch.float32, device=dev)
    q = np.asarray(quality, dtype=np.float64)
    q = (q - q.mean()) / (q.std() + 1e-8)
    w = np.exp(beta_s * q)
    # weights and multinomial stay on CPU (MPS multinomial has been unreliable)
    w = torch.as_tensor(w / w.sum(), dtype=torch.float32, device="cpu")
    gen = cpu_gen(seed)

    net = ScoreNet(Z.shape[1], hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        idx = torch.multinomial(w, batch, replacement=True, generator=gen).to(dev)
        z = Z[idx]
        u = torch.rand(batch, generator=gen, device="cpu").to(dev)
        sigma = sigma_min * (sigma_max / sigma_min) ** u          # log-uniform
        eps = torch.randn_like(z)
        pred = net(z + sigma[:, None] * eps, sigma)
        loss = ((sigma[:, None] ** 2) * (pred + eps / sigma[:, None]) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net.eval()


@torch.no_grad()
def annealed_langevin(net: ScoreNet, n: int, dz: int, steps: int = 60, eta: float = 0.01,
                      sigma_hi: float = 1.0, sigma_lo: float = 0.05, rng_seed: int = 0,
                      box: Optional[tuple] = None) -> np.ndarray:
    """GEN-CIM eq. (10): z <- z + (eta/2) s(z, sigma_t) + sqrt(eta) eps, with sigma
    annealed linearly from sigma_hi to sigma_lo. Zero decoder calls."""
    dev = next(net.parameters()).device
    g = cpu_gen(rng_seed)
    z = randn((n, dz), g, dev)
    lo_t = hi_t = None
    if box is not None:
        lo_t = torch.as_tensor(box[0], dtype=torch.float32, device=dev)
        hi_t = torch.as_tensor(box[1], dtype=torch.float32, device=dev)
    for t in range(steps):
        sigma = sigma_hi + (sigma_lo - sigma_hi) * (t / max(1, steps - 1))
        s = net(z, torch.full((n,), sigma, device=dev))
        z = z + 0.5 * eta * s + math.sqrt(eta) * randn(z.shape, g, dev)
        if lo_t is not None:
            z = torch.clamp(z, lo_t, hi_t)
    return z.detach().to("cpu").numpy()


# --------------------------------------------------------------------- flow prior

class Coupling(nn.Module):
    def __init__(self, dz: int, hidden: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(dz, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * dz),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _st(self, x):
        s, t = self.net(x * self.mask).chunk(2, dim=-1)
        s = torch.tanh(s) * 2.0
        return s * (1 - self.mask), t * (1 - self.mask)

    def forward(self, z):                       # data -> base
        s, t = self._st(z)
        u = z * self.mask + (1 - self.mask) * ((z - t) * torch.exp(-s))
        return u, -s.sum(-1)

    def inverse(self, u):                       # base -> data
        s, t = self._st(u)
        return u * self.mask + (1 - self.mask) * (u * torch.exp(s) + t)


class FlowPrior(nn.Module):
    def __init__(self, dz: int, layers: int = 6, hidden: int = 256):
        super().__init__()
        masks = []
        for i in range(layers):
            m = torch.zeros(dz)
            m[i % 2::2] = 1.0
            masks.append(m)
        self.layers = nn.ModuleList([Coupling(dz, hidden, m) for m in masks])
        self.dz = dz

    def log_prob(self, z):
        ld = torch.zeros(len(z), device=z.device)
        for layer in self.layers:
            z, d = layer(z)
            ld = ld + d
        base = -0.5 * (z ** 2).sum(-1) - 0.5 * self.dz * math.log(2 * math.pi)
        return base + ld

    @torch.no_grad()
    def sample(self, n: int, gen: Optional[torch.Generator] = None):
        dev = next(self.parameters()).device
        u = randn((n, self.dz), gen or cpu_gen(0), dev)
        for layer in reversed(self.layers):
            u = layer.inverse(u)
        return u


def train_flow(latents: np.ndarray, epochs: int = 800, batch: int = 256, lr: float = 1e-3,
               layers: int = 6, seed: int = 0,
               device: Optional[torch.device] = None) -> FlowPrior:
    """Maximum likelihood on the (unweighted) pool latents."""
    torch.manual_seed(seed)
    dev = device or get_device()
    Z = torch.as_tensor(latents, dtype=torch.float32, device=dev)
    flow = FlowPrior(Z.shape[1], layers=layers).to(dev)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    gen = cpu_gen(seed)
    for _ in range(epochs):
        idx = torch.randint(0, len(Z), (min(batch, len(Z)),), generator=gen,
                            device="cpu").to(dev)
        loss = -flow.log_prob(Z[idx]).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
        opt.step()
    return flow.eval()
