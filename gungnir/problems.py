"""Synthetic CVRP instances and the random-keys surrogate decoder.

The random-keys decoder is a wiring check, not a stand-in for the learned
latent. In CVAE-Opt's ablation a random-keys decoder scored 23.97 on CVRP100
where the learned decoder scored 15.75.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# capacity by customer count (Kool et al. / CVAE-Opt generator)
CAPACITY = {20: 30.0, 50: 40.0, 100: 50.0, 125: 55.0, 150: 60.0}


@dataclass
class CVRPInstance:
    coords: np.ndarray      # [n+1, 2], index 0 is the depot
    demands: np.ndarray     # [n+1], demands[0] == 0
    capacity: float
    dist: np.ndarray = field(init=False)

    def __post_init__(self):
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        self.dist = np.sqrt((diff ** 2).sum(-1))

    @property
    def n(self) -> int:
        return len(self.coords) - 1


def make_instance(n: int, rng: np.random.Generator) -> CVRPInstance:
    coords = rng.random((n + 1, 2))
    demands = np.concatenate([[0.0], rng.integers(1, 10, size=n).astype(float)])
    return CVRPInstance(coords, demands, CAPACITY.get(n, 50.0))


class RandomKeysDecoder:
    """Bean (1994): argsort(z) is the customer order; a new route starts whenever
    capacity would be exceeded. Always feasible. dz = n."""

    def __init__(self, inst: CVRPInstance):
        self.inst = inst
        self.dz = inst.n

    def decode(self, Z: np.ndarray) -> np.ndarray:
        inst, D = self.inst, self.inst.dist
        B, n = Z.shape
        order = np.argsort(Z, axis=1) + 1
        cost = np.zeros(B)
        load = np.zeros(B)
        cur = np.zeros(B, dtype=np.int64)
        for t in range(n):
            nxt = order[:, t]
            d = inst.demands[nxt]
            over = (load + d) > inst.capacity + 1e-9
            cost += np.where(over, D[cur, 0], 0.0)
            cur = np.where(over, 0, cur)
            load = np.where(over, 0.0, load)
            cost += D[cur, nxt]
            load += d
            cur = nxt
        cost += D[cur, 0]
        return cost


def fit_latent_box(latents: np.ndarray, coverage: float = 0.99):
    """Per-dimension bounds covering `coverage` of encoded latents."""
    lo = np.quantile(latents, (1 - coverage) / 2, axis=0)
    hi = np.quantile(latents, 1 - (1 - coverage) / 2, axis=0)
    return lo.astype(np.float32), hi.astype(np.float32)
