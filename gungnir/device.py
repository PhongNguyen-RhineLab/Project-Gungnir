"""Device handling (cpu / cuda / cuda:N / mps) shared by every module.

Random draws that seed the search (Langevin noise, flow samples, pool indices)
come from CPU generators and are moved to the device, because MPS rejects CPU
generators in device-side factory calls. This keeps those draws identical across
devices. It does NOT make whole runs bit-identical across devices: device kernels
round differently, and DSM training noise (randn_like) is drawn on the device.

MPS fallback: PyTorch reads PYTORCH_ENABLE_MPS_FALLBACK once, when torch is
imported (aten/src/ATen/mps/MPSFallback.mm). gungnir/__init__.py sets it before
any torch import; `fallback_active()` reports whether that actually happened.
"""

from __future__ import annotations

import os
import sys

import torch

_DEVICE: torch.device | None = None
_FALLBACK_AT_IMPORT = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0")


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def _mps_built() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_built())


def fallback_active() -> bool:
    """True if the MPS CPU fallback was enabled when torch was imported."""
    return _FALLBACK_AT_IMPORT


def resolve_device(spec: str | torch.device = "auto") -> torch.device:
    """'auto' picks cuda, then mps, then cpu. Explicit requests are validated so a
    typo or a missing backend fails at startup, not hours into a run."""
    if isinstance(spec, torch.device):
        return spec
    spec = (spec or "auto").strip().lower()

    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if spec == "cpu":
        return torch.device("cpu")

    if spec == "cuda" or spec.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("cuda requested but torch.cuda.is_available() is False "
                               f"(torch {torch.__version__}, built for cuda "
                               f"{torch.version.cuda or 'none'})")
        idx = int(spec.split(":", 1)[1]) if ":" in spec else torch.cuda.current_device()
        n = torch.cuda.device_count()
        if not 0 <= idx < n:
            raise RuntimeError(f"{spec} requested but only {n} cuda device(s) visible "
                               f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})")
        return torch.device("cuda", idx)

    if spec == "mps":
        if not _mps_available():
            why = ("this torch build has no MPS support" if not _mps_built()
                   else "MPS is built but not available (needs Apple Silicon and macOS 12.3+)")
            raise RuntimeError(f"mps requested but {why}")
        if not _FALLBACK_AT_IMPORT:
            print("warning: PYTORCH_ENABLE_MPS_FALLBACK was not set before torch was "
                  "imported, so ops without an MPS kernel will raise instead of falling "
                  "back to CPU. Run via `python -m gungnir` or export the variable.",
                  file=sys.stderr)
        return torch.device("mps")

    raise ValueError(f"unknown device '{spec}'; use auto | cpu | cuda | cuda:N | mps")


def set_device(spec: str | torch.device = "auto") -> torch.device:
    """Set the default device for priors. Call once at startup."""
    global _DEVICE
    _DEVICE = resolve_device(spec)
    if _DEVICE.type == "cuda":
        torch.cuda.set_device(_DEVICE)
    return _DEVICE


def get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = resolve_device("auto")
    return _DEVICE


def cpu_gen(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(int(seed))


def randn(shape, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.randn(*shape, generator=gen, device="cpu").to(device)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def device_report(device: torch.device) -> str:
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        return (f"cuda:{device.index}: {p.name}, {p.total_memory / 2**30:.1f} GiB, "
                f"torch {torch.__version__} / cuda {torch.version.cuda}")
    if device.type == "mps":
        return (f"mps: Apple Silicon, torch {torch.__version__}, "
                f"cpu fallback {'on' if _FALLBACK_AT_IMPORT else 'OFF'}")
    return f"cpu: {os.cpu_count()} cores, torch {torch.__version__}"


def available() -> dict:
    """What this machine can run, for `python -m gungnir device`."""
    return {
        "cpu": True,
        "cuda": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available() else [],
        "mps_built": _mps_built(),
        "mps_available": _mps_available(),
        "mps_fallback": _FALLBACK_AT_IMPORT,
    }


def smoke_test(device: torch.device) -> str:
    """Train a tiny ScoreNet and flow for a few steps and sample from both on
    `device`. Catches missing kernels before a long run does."""
    import time

    import numpy as np

    from .priors import annealed_langevin, train_flow, train_scorenet
    t = time.time()
    rng = np.random.default_rng(0)
    lat = rng.standard_normal((256, 100)).astype(np.float32)
    sn = train_scorenet(lat, -np.abs(lat).sum(1), epochs=5, device=device)
    z = annealed_langevin(sn, 64, 100, steps=5, box=(-np.ones(100, np.float32) * 2,
                                                     np.ones(100, np.float32) * 2))
    fl = train_flow(lat, epochs=5, device=device)
    s = fl.sample(64, cpu_gen(0)).to("cpu").numpy()
    synchronize(device)
    ok = np.isfinite(z).all() and np.isfinite(s).all()
    return f"{'ok' if ok else 'NON-FINITE OUTPUT'} on {device} in {time.time() - t:.2f}s"