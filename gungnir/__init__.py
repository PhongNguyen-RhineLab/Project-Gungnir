"""
gungnir
=======
Does a learned prior over a frozen CVAE latent space improve latent search on
CVRP, at equal decode budget?

Every arm is measured in decoder calls, never wall clock. Prior-based arms pay
for the decodes that built their training pool (see `BudgetedDecoder.charge_setup`).

Layout
------
  device.py       cuda / mps / cpu resolution, CPU-side generators
  budget.py       Decoder protocol, BudgetedDecoder (hard budget + anytime trace)
  problems.py     synthetic CVRP instances, random-keys surrogate decoder
  priors.py       ScoreNet + DSM + annealed Langevin, RealNVP flow prior
  samplers.py     the five arms and the ARMS registry
  experiment.py   build_pool, run_arm, summarise, landscape check, tuner, selftest
  cvae_opt.py     adapter for the pretrained CVAE-Opt decoder
  runner.py       the real-decoder experiment (formerly run_real.py)
  analysis.py     paired statistics and anytime curves (formerly analyze_results.py)
  calibration.py  numba random-keys DE yardstick (formerly calib2.py + sweep.py)
  cli.py          `python -m gungnir <command>`
"""

import os as _os

# Must happen before torch is imported anywhere: PyTorch reads this variable once,
# at import, to decide whether ops without an MPS kernel fall back to CPU or raise.
# Harmless on machines without MPS. Export PYTORCH_ENABLE_MPS_FALLBACK=0 to opt out.
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.2.1"