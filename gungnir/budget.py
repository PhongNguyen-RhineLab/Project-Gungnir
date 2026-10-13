"""Budget accounting. Every sampler must spend decodes through BudgetedDecoder."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np


class Decoder(Protocol):
    dz: int

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """[B, dz] float32 latents -> [B] costs."""
        ...


class BudgetedDecoder:
    """Wraps a decoder, hard-caps total decodes, and records the anytime curve.

    Batches that would overspend are truncated; the truncated tail gets cost +inf
    so no arm can quietly exceed its budget.

    `trace` holds (decodes_used, best_so_far) after every call. When setup decodes
    are charged, one point (setup, +inf) is recorded first: at that point the arm
    has spent decodes and still has no solution of its own.
    """

    def __init__(self, decoder: Decoder, budget: int):
        self.decoder = decoder
        self.budget = int(budget)
        self.used = 0
        self.setup_charged = 0
        self.best = math.inf
        self.best_z: Optional[np.ndarray] = None
        self.trace: List[Tuple[int, float]] = []
        self.diagnostics: Dict[str, float] = {}

    @property
    def dz(self) -> int:
        return self.decoder.dz

    @property
    def exhausted(self) -> bool:
        return self.used >= self.budget

    def remaining(self) -> int:
        return max(0, self.budget - self.used)

    def charge_setup(self, evals: int, amortize_over: int = 1) -> int:
        """Charge the decodes that built an arm's prior against its budget.

        `amortize_over` is the number of instances the SAME prior is reused on.
        It is only honest when the pool was not built on the instance being
        searched; see runner.py --pool-mode.
        """
        k = int(math.ceil(max(0, evals) / max(1, amortize_over)))
        if k >= self.budget:
            raise ValueError(f"setup cost {k} >= budget {self.budget}: the arm would "
                             f"never decode a single search point")
        if k:
            self.used += k
            self.setup_charged += k
            self.trace.append((self.used, self.best))
        return k

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float32)
        k = min(len(Z), self.remaining())
        if k == 0:
            return np.full(len(Z), math.inf, dtype=np.float64)
        costs = np.asarray(self.decoder.decode(Z[:k]), dtype=np.float64)
        self.used += k
        i = int(np.argmin(costs))
        if costs[i] < self.best:
            self.best = float(costs[i])
            self.best_z = Z[i].copy()
        self.trace.append((self.used, self.best))
        if k < len(Z):
            costs = np.concatenate([costs, np.full(len(Z) - k, math.inf)])
        return costs


def best_at(trace, evals: int, offset: int = 0) -> float:
    """Best cost after `evals` decodes. With `offset`, the axis is decodes spent
    beyond `offset` (use the arm's setup charge to get search decodes)."""
    target = evals + offset
    best = math.inf
    for e, b in trace:
        if e > target:
            break
        best = b
    return best
