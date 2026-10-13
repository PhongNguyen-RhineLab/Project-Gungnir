"""Adapter for the pretrained CVAE-Opt model (github.com/ahottung/CVAE-Opt).

* The search box is [-Z_bound, Z_bound]^dz from the checkpoint (2.073 for
  cvrp_100), not N(0, I).
* dz is 100 for every problem size.
* The decoder is stateful: reset_decoder(bs) whenever the batch size changes,
  with the instance tensor pre-expanded to that batch size.
* Decoding is greedy under no_grad, matching the published numbers.
"""

from __future__ import annotations

import os
import pickle
import sys
import time
import types
from typing import List

import numpy as np
import torch

from .device import resolve_device, synchronize

MODELS = {
    (100, "CVRP"): "cvrp_100_model_78580.pt",
    (50, "CVRP"): "cvrp_50_model_96631.pt",
    (20, "CVRP"): "cvrp_20_model_30047.pt",
}
TESTSET = {100: "vrp100_test_small_seed1235.pkl",
           50: "vrp50_test_small_seed1235.pkl",
           20: "vrp20_test_small_seed1235.pkl"}
# LKH3 means on the full 1000-instance sets (CVAE-Opt paper). Display only.
LKH3 = {20: 6.14, 50: 10.38, 100: 15.54}

_REQUIRED = ("VAE_8.py", "cvrp.py", "tsp.py", "utils.py")


def preflight(repo: str) -> str:
    """Resolve and validate the CVAE-Opt checkout before touching sys.path, so a
    wrong --repo fails with an actionable message instead of
    `ModuleNotFoundError: No module named 'VAE_8'`."""
    abs_repo = os.path.abspath(os.path.expanduser(repo))
    cwd = os.getcwd()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # fall back to the usual locations only when --repo was left at its default;
    # an explicit wrong path must fail, not silently load a different checkout
    if not os.path.isdir(abs_repo) and os.path.normpath(repo) == "CVAE-Opt":
        for cand in (os.path.join(here, "CVAE-Opt"), os.path.join(cwd, "CVAE-Opt"),
                     os.path.join(here, "..", "CVAE-Opt")):
            cand = os.path.abspath(cand)
            if os.path.isfile(os.path.join(cand, "VAE_8.py")):
                return cand
    if not os.path.isdir(abs_repo):
        raise FileNotFoundError(
            f"CVAE-Opt repo not found.\n"
            f"  --repo resolved to : {abs_repo}\n"
            f"  working directory  : {cwd}\n"
            f"  package lives in   : {here}\n\n"
            f"    git clone --depth 1 https://github.com/ahottung/CVAE-Opt.git\n"
            f"    python -m gungnir run --repo /absolute/path/to/CVAE-Opt ...\n\n"
            f"--repo is relative to the WORKING directory (in PyCharm usually the "
            f"project root). Use an absolute path if unsure.")
    missing = [f for f in _REQUIRED if not os.path.isfile(os.path.join(abs_repo, f))]
    if missing:
        listing = sorted(os.listdir(abs_repo))
        raise FileNotFoundError(
            f"{abs_repo} exists but is not a CVAE-Opt checkout.\n"
            f"  missing : {', '.join(missing)}\n"
            f"  contains: {', '.join(listing[:12])}{' ...' if len(listing) > 12 else ''}\n"
            f"If the clone was interrupted, delete the folder and re-clone.")
    return abs_repo


def load_model(repo: str, size: int, device="auto", batch_size: int = 600, problem="CVRP"):
    """Returns (model, cfg, z_bound, meta). `repo` must already be preflighted."""
    dev = resolve_device(device)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from VAE_8 import VAE_8  # noqa: E402

    path = os.path.join(repo, "models", MODELS[(size, problem)])
    if not os.path.isfile(path):
        avail = os.path.join(repo, "models")
        have = sorted(os.listdir(avail)) if os.path.isdir(avail) else []
        raise FileNotFoundError(f"checkpoint not found: {path}\n"
                                f"  available: {', '.join(have) or '(none)'}")
    meta = torch.load(path, map_location="cpu", weights_only=False)
    # CVAE-Opt passes config.device straight to torch factories: give it a string.
    cfg = types.SimpleNamespace(device=str(dev), problem=meta["problem"],
                                problem_size=meta["problem_size"], search_space_size=100,
                                search_batch_size=batch_size, output_path=repo)
    model = VAE_8(cfg).to(dev)
    model.load_state_dict(meta["parameters"])
    model.eval()
    return model, cfg, float(meta["Z_bound"]), meta


def read_cvrp_instances(path: str, problem_size: int) -> List[np.ndarray]:
    """Kool-format pickle -> [n+1, 4] arrays of (x, y, 1.0, demand/capacity)."""
    with open(path, "rb") as f:
        raw = pickle.load(f)
    out = []
    for depot, locs, dem, cap in raw:
        a = np.zeros((problem_size + 1, 4))
        a[0, :2] = depot
        a[1:, :2] = locs
        a[:, 2] = 1.0
        a[1:, 3] = np.array(dem) / cap
        out.append(a)
    return out


def test_instances(repo: str, size: int) -> List[np.ndarray]:
    return read_cvrp_instances(os.path.join(repo, "instances", "cvrp", "test", TESTSET[size]), size)


class CVAEOptDecoder:
    """z in [-Z_bound, Z_bound]^100 -> feasible tour (greedy) -> tour length."""

    def __init__(self, model, cfg, instance: np.ndarray, device="auto", batch_size: int = 600):
        if cfg.output_path not in sys.path:
            sys.path.insert(0, cfg.output_path)
        import cvrp
        import tsp
        self.model, self.cfg = model, cfg
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.dz = cfg.search_space_size
        self.cost_fn = cvrp.tours_length if cfg.problem == "CVRP" else tsp.tours_length
        # float32 explicitly: MPS has no float64
        self._inst1 = torch.as_tensor(np.asarray(instance, dtype=np.float32)
                                      ).unsqueeze(0).to(self.device)
        self._cur_bs = None
        self._inst = None

    def _prepare(self, bs: int):
        if bs != self._cur_bs:
            self._inst = self._inst1.expand(bs, -1, -1).contiguous()
            self.model.reset_decoder(bs, self.cfg)
            self._cur_bs = bs

    @torch.no_grad()
    def decode(self, Z: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(Z), self.batch_size):
            zb = np.asarray(Z[i:i + self.batch_size], dtype=np.float32)
            self._prepare(len(zb))
            _, tour_idx, _ = self.model.decode(self._inst, torch.as_tensor(zb).to(self.device),
                                               self.cfg)
            out.append(self.cost_fn(self._inst, tour_idx).to("cpu").numpy())
        return np.concatenate(out).astype(np.float64)

    def box(self, z_bound: float):
        lo = np.full(self.dz, -z_bound, dtype=np.float32)
        return lo, -lo

    def benchmark(self, n: int = None) -> float:
        """Decodes per second at the current batch size."""
        n = n or self.batch_size
        Z = np.zeros((n, self.dz), dtype=np.float32)
        self.decode(Z[: min(8, n)])          # warm-up
        synchronize(self.device)
        t = time.time()
        self.decode(Z)
        synchronize(self.device)
        return n / (time.time() - t)
