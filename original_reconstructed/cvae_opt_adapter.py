"""
cvae_opt_adapter.py
===================
Wraps the pretrained CVAE-Opt model (github.com/ahottung/CVAE-Opt) as a
`Decoder` for latent_search.py, so the five proposal-distribution arms can be
compared on the real learned latent space instead of the random-keys stand-in.

Usage
-----
    git clone --depth 1 https://github.com/ahottung/CVAE-Opt.git
    python3 run_real.py --repo ./CVAE-Opt --size 100 --instances 5 --budget 180600

Things this file gets right that a hand-rolled wrapper usually gets wrong
------------------------------------------------------------------------
* The search box is NOT N(0, I). Every checkpoint carries `Z_bound`, and
  CVAE-Opt searches the box [-Z_bound, +Z_bound]^dz. For cvrp_100 that is
  2.073. Sampling from a unit Gaussian instead would put roughly 30% of the
  mass outside the region the decoder was calibrated on.
* dz is 100 for every problem size, not the number of customers.
* The decoder is stateful: `reset_decoder(batch_size)` must be called whenever
  the batch size changes, and the instance tensor must be pre-expanded to that
  batch size.
* Decoding is greedy and under no_grad, matching the published numbers.
"""

from __future__ import annotations

import os
import pickle
import sys
import types
from typing import List

import numpy as np
import torch

import latent_search as L


def _make_config(repo: str, device: str, problem: str, problem_size: int,
                 search_space_size: int, batch_size: int):
    """CVAE-Opt reads settings off an argparse Namespace; build one directly."""
    return types.SimpleNamespace(
        device=device, problem=problem, problem_size=problem_size,
        search_space_size=search_space_size, search_batch_size=batch_size,
        output_path=repo,
    )


#: files that must be present in the CVAE-Opt checkout for the adapter to work
_REQUIRED = ("VAE_8.py", "cvrp.py", "tsp.py", "utils.py")


def _preflight(repo: str) -> str:
    """Resolve and validate the CVAE-Opt checkout before touching sys.path.

    Without this, a wrong --repo surfaces as `ModuleNotFoundError: No module
    named 'VAE_8'` several frames deep, which points at the import rather than
    at the actual problem.
    """
    abs_repo = os.path.abspath(os.path.expanduser(repo))
    cwd = os.getcwd()
    here = os.path.dirname(os.path.abspath(__file__))

    if not os.path.isdir(abs_repo):
        # Look in the usual places before giving up.
        for cand in (os.path.join(here, "CVAE-Opt"),
                     os.path.join(cwd, "CVAE-Opt"),
                     os.path.join(here, "..", "CVAE-Opt")):
            cand = os.path.abspath(cand)
            if os.path.isfile(os.path.join(cand, "VAE_8.py")):
                return cand
        raise FileNotFoundError(
            f"CVAE-Opt repo not found.\n"
            f"  --repo resolved to : {abs_repo}\n"
            f"  working directory  : {cwd}\n"
            f"  this script lives in: {here}\n\n"
            f"Clone it and point --repo at the checkout:\n"
            f"    git clone --depth 1 https://github.com/ahottung/CVAE-Opt.git\n"
            f"    python3 run_real.py --repo /absolute/path/to/CVAE-Opt ...\n\n"
            f"Note: --repo is relative to the WORKING directory, which in "
            f"PyCharm is often the project root rather than the script's "
            f"folder. Use an absolute path if unsure.")

    missing = [f for f in _REQUIRED if not os.path.isfile(os.path.join(abs_repo, f))]
    if missing:
        listing = sorted(os.listdir(abs_repo))[:12]
        raise FileNotFoundError(
            f"{abs_repo} exists but is not a CVAE-Opt checkout.\n"
            f"  missing: {', '.join(missing)}\n"
            f"  contains: {', '.join(listing)}"
            f"{' ...' if len(os.listdir(abs_repo)) > 12 else ''}\n\n"
            f"If the clone was interrupted, delete the folder and re-clone.")
    return abs_repo


def load_model(repo: str, model_path: str, device: str = "auto",
               batch_size: int = 600):
    """Returns (model, config, z_bound, meta). Adds `repo` to sys.path.

    `device` accepts 'auto', 'cuda', 'cuda:N', 'mps', or 'cpu'.

    MPS notes. CVAE-Opt has no hardcoded .cuda() calls -- it threads
    config.device through everywhere -- so the model itself carries over. Two
    things still bite:
      * The decoder's masking path uses ops (nonzero, scatter on bool) that
        may lack MPS kernels depending on torch version. resolve_device sets
        PYTORCH_ENABLE_MPS_FALLBACK=1 so those fall back to CPU rather than
        raising partway through a long run.
      * MPS has no float64. The checkpoints are float32 and the cost function
        stays float32, so this is fine, but do not cast instances to double.
    """
    dev = L.resolve_device(device)
    repo = _preflight(repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from VAE_8 import VAE_8  # noqa: E402  (import must follow the path insert)

    model_path = os.path.abspath(os.path.expanduser(model_path))
    if not os.path.isfile(model_path):
        avail = os.path.join(repo, "models")
        have = sorted(os.listdir(avail)) if os.path.isdir(avail) else []
        raise FileNotFoundError(
            f"checkpoint not found: {model_path}\n"
            f"  available in {avail}: {', '.join(have) or '(none)'}")

    meta = torch.load(model_path, map_location="cpu", weights_only=False)
    # CVAE-Opt stores config.device as a string and passes it straight to
    # torch factory calls, so hand it the resolved string, not a device object.
    cfg = _make_config(repo, str(dev), meta["problem"], meta["problem_size"],
                       search_space_size=100, batch_size=batch_size)
    model = VAE_8(cfg).to(dev)
    model.load_state_dict(meta["parameters"])
    model.eval()
    return model, cfg, float(meta["Z_bound"]), meta


def read_cvrp_instances(path: str, problem_size: int) -> List[np.ndarray]:
    """Kool-format pickle -> [n+1, 4] arrays of (x, y, load, normalised demand)."""
    with open(path, "rb") as f:
        raw = pickle.load(f)
    out = []
    for depot, locs, dem, cap in raw:
        a = np.zeros((problem_size + 1, 4))
        a[0, :2] = depot
        a[1:, :2] = locs
        a[:, 2] = 1.0
        a[1:, 3] = np.array(dem) / cap
        a[0, 3] = 0.0
        out.append(a)
    return out


class CVAEOptDecoder:
    """z in [-Z_bound, Z_bound]^100  ->  feasible CVRP tour  ->  tour length."""

    def __init__(self, model, cfg, instance: np.ndarray, device: str = "auto",
                 batch_size: int = 600):
        import cvrp, tsp  # noqa: E402
        self.model, self.cfg = model, cfg
        self.device = L.resolve_device(device) if isinstance(device, str) else device
        self.batch_size = batch_size
        self.dz = cfg.search_space_size
        self.cost_fn = cvrp.tours_length if cfg.problem == "CVRP" else tsp.tours_length
        # float32 explicitly: MPS has no float64 and np arrays default to it.
        self._inst1 = torch.as_tensor(
            np.asarray(instance, dtype=np.float32)).unsqueeze(0).to(self.device)
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
            zt = torch.as_tensor(zb).to(self.device)
            _, tour_idx, _ = self.model.decode(self._inst, zt, self.cfg)
            out.append(self.cost_fn(self._inst, tour_idx).to("cpu").numpy())
        return np.concatenate(out).astype(np.float64)

    def box(self, z_bound: float):
        """The search box every arm must be clipped to."""
        lo = np.full(self.dz, -z_bound, dtype=np.float32)
        return lo, -lo

    def benchmark(self, n: int = None) -> float:
        """Decodes per second. Run this before committing to a budget.

        On MPS especially, a batch size that is too large can be slower than a
        moderate one, and any op that falls back to CPU shows up here as a
        cliff. Sweep --batch over 128/300/600/1200 once rather than guessing.
        """
        import time
        n = n or self.batch_size
        Z = np.zeros((n, self.dz), dtype=np.float32)
        self.decode(Z[: min(8, n)])                    # warm up / compile
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()
        t = time.time()
        self.decode(Z)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()
        return n / (time.time() - t)
