"""Experiment plumbing shared by the self-test and the real-decoder runner."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from .budget import BudgetedDecoder, Decoder
from .device import get_device
from .priors import FlowPrior, ScoreNet, train_flow, train_scorenet
from .problems import RandomKeysDecoder, make_instance
from .samplers import ARMS, PRIOR_ARMS, clip, sampler_metropolis


# --------------------------------------------------------------------- priors

def build_pool(d: BudgetedDecoder, rng, size: int, keep_frac: float = 0.5, box=None):
    """Sample `size` latents from N(0, I) clipped to the box, decode THROUGH the
    budgeted decoder, keep the best `keep_frac` (GEN-CIM Phase 2).
    Returns (latents, quality) with quality = -cost."""
    Z = clip(rng.standard_normal((size, d.dz)).astype(np.float32), box)
    c = d(Z)
    keep = np.argsort(c)[: max(1, int(size * keep_frac))]
    return Z[keep], -c[keep]


@dataclass
class Priors:
    scorenet: Optional[ScoreNet] = None
    flow: Optional[FlowPrior] = None
    pool_evals: int = 0          # decodes spent building the pool these were fit on

    @classmethod
    def fit(cls, lat, qual, arms: Iterable[str], score_epochs: int, flow_epochs: int,
            seed: int, pool_evals: int) -> "Priors":
        """Train only the priors the requested arms use."""
        need = {ARMS[a].prior for a in arms if ARMS[a].needs_prior}
        sn = train_scorenet(lat, qual, epochs=score_epochs, seed=seed) if "scorenet" in need else None
        fl = train_flow(lat, epochs=flow_epochs, seed=seed) if "flow" in need else None
        return cls(sn, fl, pool_evals)


def arm_configs(arms: Iterable[str], pop: int, box=None, priors: Optional[Priors] = None,
                chains: int = 8, lam: float = 5.0, sigma_prop: float = 0.1) -> Dict[str, dict]:
    cfg = {}
    for a in arms:
        if a in ("random", "de"):
            cfg[a] = dict(pop=pop, box=box)
        elif a == "flow_de":
            cfg[a] = dict(pop=pop, box=box, flow=priors.flow)
        elif a == "langevin_de":
            cfg[a] = dict(pop=pop, box=box, scorenet=priors.scorenet)
        elif a == "metropolis":
            cfg[a] = dict(chains=chains, lam=lam, sigma_prop=sigma_prop, box=box)
    return cfg


# --------------------------------------------------------------------- running

def run_arm(name: str, decoder: Decoder, budget: int, seed: int, setup_evals: int = 0,
            amortize_over: int = 1, **cfg) -> dict:
    """Run one arm until `budget` TOTAL decodes are spent, setup included."""
    arm = ARMS[name]
    if arm.needs_prior and setup_evals == 0 and amortize_over == 1:
        raise ValueError(
            f"arm '{name}' uses a prior fitted on decoded solutions but declared "
            f"setup_evals=0. Pass the pool size (or amortize_over=n for a prior reused "
            f"across n instances). An unpriced pool hands the arm free decodes.")
    d = BudgetedDecoder(decoder, budget)
    d.charge_setup(setup_evals, amortize_over)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    arm.fn(d, rng, **cfg)
    return {"arm": name, "seed": seed, "best": d.best, "evals": d.used,
            "setup": d.setup_charged, "seconds": time.time() - t0,
            "trace": d.trace, "diagnostics": d.diagnostics}


def setup_for(arm: str, priors: Optional[Priors]) -> int:
    return priors.pool_evals if (arm in PRIOR_ARMS and priors is not None) else 0


def instance_line(results: List[dict], inst, arms: List[str]) -> str:
    """Per-instance line, mean over seeds (the same statistic the summary uses)."""
    parts = []
    for a in arms:
        v = [r["best"] for r in results if r["instance"] == inst and r["arm"] == a]
        if v:
            parts.append(f"{a}={np.mean(v):.3f}")
    return f"  instance {inst}: " + "  ".join(parts)


def summarise(results: List[dict], arms: List[str]) -> str:
    """Per-instance reference = best mean-over-seeds of any arm. Gap is relative
    to that reference, not to an optimum. wins counts arms tied at the reference."""
    by_inst: Dict = {}
    setup: Dict[str, int] = {}
    for r in results:
        by_inst.setdefault(r["instance"], {}).setdefault(r["arm"], []).append(r["best"])
        setup[r["arm"]] = r.get("setup", 0)
    means = {i: {a: float(np.mean(v)) for a, v in m.items()} for i, m in by_inst.items()}
    ref = {i: min(m.values()) for i, m in means.items()}

    lines = [f"{'arm':<14}{'mean cost':>11}{'mean gap':>10}{'std gap':>9}{'wins':>7}{'setup':>9}",
             "-" * 60]
    rows = []
    for arm in arms:
        gaps, costs, wins = [], [], 0
        for i, m in means.items():
            if arm not in m:
                continue
            costs.append(m[arm])
            gaps.append(100.0 * (m[arm] / ref[i] - 1.0))
            wins += abs(m[arm] - ref[i]) < 1e-9
        if costs:
            rows.append((arm, np.mean(costs), np.mean(gaps), np.std(gaps), wins))
    for arm, c, g, s, w in sorted(rows, key=lambda r: r[2]):
        lines.append(f"{arm:<14}{c:>11.4f}{g:>9.3f}%{s:>8.3f}%{w:>7d}{setup.get(arm, 0):>9d}")
    return "\n".join(lines)


def acceptance_note(results: List[dict]) -> Optional[str]:
    ar = [r["diagnostics"]["accept_rate"] for r in results
          if r["arm"] == "metropolis" and "accept_rate" in r.get("diagnostics", {})]
    if not ar:
        return None
    m = float(np.mean(ar))
    flag = "" if 0.2 <= m <= 0.5 else "   <-- outside 0.2-0.5, retune lam/chains"
    return f"metropolis mean acceptance {m:.3f}{flag}"


def dump_results(results: List[dict], path: str, keep_traces: bool = False):
    drop = () if keep_traces else ("trace",)
    with open(path, "w") as f:
        json.dump([{k: v for k, v in r.items() if k not in drop} for r in results], f, indent=1)


# --------------------------------------------------------------------- stage 0

def landscape_check(decoder: Decoder, rng, n_ref=1000, per_shell=100,
                    radii=(0.1, 0.2, 0.5, 2.0, 5.0), box=None):
    """CVAE-Opt's search-space structure test: best of n_ref random latents, then
    mean cost penalty on shells of increasing radius around it. Monotone rise means
    neighbouring latents decode to similar quality; flat means no local structure."""
    Z = clip(rng.standard_normal((n_ref, decoder.dz)).astype(np.float32), box)
    c = decoder.decode(Z)
    z_ref, c_ref = Z[int(np.argmin(c))], float(c.min())
    out = []
    for r in radii:
        eps = rng.standard_normal((per_shell, decoder.dz))
        eps /= np.linalg.norm(eps, axis=1, keepdims=True)
        Zs = clip((z_ref + r * eps).astype(np.float32), box)
        out.append((r, float(np.mean(decoder.decode(Zs)) - c_ref)))
    return c_ref, out


def print_landscape(decoder, rng, **kw):
    c_ref, shells = landscape_check(decoder, rng, **kw)
    print(f"landscape check | reference cost {c_ref:.3f}")
    for r, d in shells:
        print(f"  radius {r:>5.2f}   mean cost penalty {d:+7.3f}  {'#' * int(max(0.0, d) * 4)}")
    span = shells[-1][1] - shells[0][1]
    smooth = span > 0.25 * abs(shells[-1][1]) + 1e-9
    print(f"  verdict: {'SMOOTH (local search viable)' if smooth else 'FLAT (local samplers will not help)'}")
    return shells


# --------------------------------------------------------------------- tuning

def tune_metropolis(decoder, budget, lams=(1, 5, 20, 100), chain_counts=(8, 20, 50, 200),
                    seeds=2, verbose=True, **cfg):
    """Joint sweep of chains x lam. Run on validation instances, not test."""
    rows = []
    for chains in chain_counts:
        for lam in lams:
            best, acc = [], []
            for s in range(seeds):
                d = BudgetedDecoder(decoder, budget)
                sampler_metropolis(d, np.random.default_rng(s), chains=chains, lam=lam, **cfg)
                best.append(d.best)
                acc.append(d.diagnostics["accept_rate"])
            rows.append(dict(chains=chains, iters=budget // chains, lam=lam,
                             best=float(np.mean(best)), accept=float(np.mean(acc))))
            if verbose:
                r = rows[-1]
                print(f"  chains={chains:<5} iters={r['iters']:<6} lam={lam:<7g} "
                      f"best={r['best']:.4f}  accept={r['accept']:.3f}")
    best_row = min(rows, key=lambda r: r["best"])
    if verbose:
        print(f"  -> best cell: chains={best_row['chains']} lam={best_row['lam']}")
    return rows, best_row


# --------------------------------------------------------------------- self-test

def selftest(n=50, instances=5, seeds=3, budget=30000, pop=300, lam=5.0, pool_size=4000,
             arms=None, out=None, keep_traces=False):
    """All arms on the random-keys surrogate. Validates wiring only."""
    arms = arms or list(ARMS)
    print(f"self-test | n={n} instances={instances} seeds={seeds} budget={budget} "
          f"device={get_device()}")
    print("NOTE: random-keys decoder. Validates the samplers, does NOT predict\n"
          "      behaviour on a learned CVAE latent space.\n")
    print_landscape(RandomKeysDecoder(make_instance(n, np.random.default_rng(1000))),
                    np.random.default_rng(0))
    print()

    results = []
    for inst_id in range(instances):
        rng = np.random.default_rng(1000 + inst_id)
        dec = RandomKeysDecoder(make_instance(n, rng))
        priors = None
        if PRIOR_ARMS & set(arms):
            lat, qual = build_pool(BudgetedDecoder(dec, pool_size), rng, pool_size)
            priors = Priors.fit(lat, qual, arms, score_epochs=300, flow_epochs=600,
                                seed=inst_id, pool_evals=pool_size)
        cfg = arm_configs(arms, pop, None, priors, chains=max(8, budget // 1500), lam=lam)
        for seed in range(seeds):
            for arm in arms:
                r = run_arm(arm, dec, budget, seed=seed * 97 + inst_id,
                            setup_evals=setup_for(arm, priors), **cfg[arm])
                r["instance"] = inst_id
                results.append(r)
        print(instance_line(results, inst_id, arms))

    print("\n" + summarise(results, arms))
    note = acceptance_note(results)
    if note:
        print("\n" + note)
    if out:
        dump_results(results, out, keep_traces)
        print(f"\nwrote {out}")
    return results
