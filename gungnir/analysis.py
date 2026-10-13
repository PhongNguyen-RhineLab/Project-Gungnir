"""Paired analysis of result JSON files (formerly analyze_results.py).

    python -m gungnir analyze confirm.json --baseline flow_de
    python -m gungnir analyze confirm.json --baseline de --curve

Unit of replication
-------------------
Seeds within an instance share one prior pool, so they are not independent for
the prior arms. The default `--unit instance` averages seeds within each instance
before pairing (n = instances). `--unit cell` pairs every (instance, seed) cell,
which is what the old script did and overstates n.

Anytime curve (--curve)
-----------------------
Needs traces (`--keep-traces` at run time). Reports the paired delta vs the
baseline at fixed checkpoints on two axes:
  total   decodes including setup (what the budget actually bought)
  search  decodes after setup (isolates search quality from setup cost)
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from math import comb

import numpy as np

from .budget import best_at

COST_KEYS = ("best", "best_cost", "cost", "obj", "objective")
INST_KEYS = ("instance", "inst", "instance_id", "inst_id")
SEED_KEYS = ("seed", "run_seed")
SETUP_KEYS = ("setup", "setup_evals", "setup_cost")


def _pick(rec, keys, default=None):
    for k in keys:
        if k in rec:
            return rec[k]
    return default


def load(path):
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        for v in raw.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                raw = v
                break
    if not isinstance(raw, list):
        sys.exit("could not find a list of result records in that file")
    return raw


# --------------------------------------------------------------------- statistics

def paired_bootstrap(d, n_boot, rng):
    d = np.asarray(d, dtype=float)
    if len(d) < 2:
        m = float(d.mean()) if len(d) else math.nan
        return m, math.nan, math.nan
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_test(d):
    """Two-sided exact sign test, ties dropped. Returns (p, wins, n)."""
    d = [x for x in d if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0, 0, 0
    wins = sum(1 for x in d if x < 0)
    k = min(wins, n - wins)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n), wins, n


def cohen_dz(d):
    d = np.asarray(d, dtype=float)
    return float(d.mean() / d.std(ddof=1)) if len(d) > 1 and d.std(ddof=1) > 0 else math.nan


def verdict(lo, hi):
    if np.isnan(hi - lo):
        return "too few pairs"
    if lo > 0 or hi < 0:
        return "DIFFERENT"
    if hi - lo < 0.5:
        return "null (tight)"
    if hi - lo < 1.5:
        return "null (loose)"
    return "UNDERPOWERED"


# --------------------------------------------------------------------- pairing

def unit_values(raw, value_fn, unit="instance"):
    """{(unit_key, arm): value}. value_fn(record) -> float or None."""
    acc = defaultdict(list)
    for r in raw:
        arm, v = r.get("arm"), value_fn(r)
        if arm is None or v is None or not np.isfinite(v):
            continue
        inst = _pick(r, INST_KEYS, 0)
        key = inst if unit == "instance" else (inst, _pick(r, SEED_KEYS, 0))
        acc[(key, arm)].append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def paired_deltas(vals, arm, base):
    """Percent deltas arm vs base over units that have both. Negative = arm better."""
    units = sorted({k[0] for k in vals}, key=str)
    return [100.0 * (vals[(u, arm)] / vals[(u, base)] - 1.0)
            for u in units if (u, arm) in vals and (u, base) in vals and vals[(u, base)] > 0]


# --------------------------------------------------------------------- report

def report(raw, baseline="de", n_boot=20000, seed=0, unit="instance", curve=False,
           checkpoints=None):
    keys = sorted({k for r in raw for k in r})
    has_trace = "trace" in keys
    print(f"records: {len(raw)}")
    print(f"keys:    {', '.join(keys)}")
    insts = {_pick(r, INST_KEYS, 0) for r in raw}
    seeds_per = defaultdict(set)
    for r in raw:
        seeds_per[_pick(r, INST_KEYS, 0)].add(_pick(r, SEED_KEYS, 0))
    print(f"design:  {len(insts)} instances x {max(len(s) for s in seeds_per.values())} seeds; "
          f"unit of replication = {unit}")
    print()

    best = lambda r: _pick(r, COST_KEYS)  # noqa: E731
    vals = unit_values(raw, best, unit)
    arms = sorted({k[1] for k in vals})
    if not arms:
        sys.exit("no 'arm' field found in the records")
    base = baseline if baseline in arms else arms[0]
    if base != baseline:
        print(f"baseline '{baseline}' not present, using '{base}'\n")

    print("raw mean cost per arm (instance variation NOT removed)")
    print(f"  {'arm':<14}{'mean':>10}{'std':>10}{'n':>6}")
    for arm in arms:
        v = [x for k, x in vals.items() if k[1] == arm]
        print(f"  {arm:<14}{np.mean(v):>10.4f}{np.std(v):>10.4f}{len(v):>6}")
    print()

    rng = np.random.default_rng(seed)
    print(f"paired delta vs '{base}', percent, negative = arm better")
    print(f"  {'arm':<14}{'mean%':>9}{'CI low':>9}{'CI high':>9}{'W/n':>8}{'p_sign':>8}"
          f"{'dz':>7}   verdict")
    print("  " + "-" * 74)
    npairs = 0
    for arm in arms:
        if arm == base:
            continue
        d = paired_deltas(vals, arm, base)
        npairs = max(npairs, len(d))
        if not d:
            print(f"  {arm:<14}{'no paired units':>40}")
            continue
        m, lo, hi = paired_bootstrap(d, n_boot, rng)
        p, w, n = sign_test(d)
        print(f"  {arm:<14}{m:>9.3f}{lo:>9.3f}{hi:>9.3f}{w:>4}/{n:<3}{p:>8.3f}"
              f"{cohen_dz(d):>7.2f}   {verdict(lo, hi)}")
    print(f"\n  {npairs} paired units")
    if npairs < 15:
        print("  warning: fewer than 15 pairs; a 'null' here means no measurement, not no effect.")

    setups = defaultdict(set)
    for r in raw:
        s = _pick(r, SETUP_KEYS)
        if s is not None:
            setups[r.get("arm")].add(int(s))
    if setups:
        print("\nsetup decodes charged per arm")
        for arm in arms:
            print(f"  {arm:<14}{sorted(setups.get(arm, {0}))}")
        bad = [x for x in ("flow_de", "langevin_de") if x in arms and setups.get(x) in (None, {0})]
        if bad:
            print(f"  WARNING: prior arms {bad} were charged zero setup; not comparable.")

    ar = [r["diagnostics"]["accept_rate"] for r in raw
          if isinstance(r.get("diagnostics"), dict) and "accept_rate" in r["diagnostics"]]
    if ar:
        m = float(np.mean(ar))
        print(f"\nmetropolis mean acceptance: {m:.3f}"
              f"{'' if 0.15 <= m <= 0.6 else '   <-- outside 0.15-0.6, retune'}")

    if has_trace:
        saturation(raw, arms)
        if curve:
            anytime_curve(raw, arms, base, unit, checkpoints)
    elif curve:
        print("\n--curve needs traces; rerun with --keep-traces")


def _traces(raw):
    return [r for r in raw if isinstance(r.get("trace"), list) and len(r["trace"]) > 1]


def saturation(raw, arms):
    """Fraction of the arm's own improvement reached by each quarter of its budget,
    measured on the decode axis from the first finite best onward."""
    tr = _traces(raw)
    if not tr:
        return
    print("\nsaturation: fraction of improvement reached at 25/50/75/100% of the decodes")
    for arm in arms:
        fr = []
        for r in (x for x in tr if x.get("arm") == arm):
            t = [(e, b) for e, b in r["trace"] if np.isfinite(b)]
            if len(t) < 2:
                continue
            e0, b0 = t[0]
            e1, b1 = t[-1]
            if b0 - b1 <= 0:
                continue
            fr.append([(b0 - best_at(t, int(e0 + q * (e1 - e0)))) / (b0 - b1)
                       for q in (0.25, 0.5, 0.75, 1.0)])
        if fr:
            print(f"  {arm:<14}" + "".join(f"{x:>8.2f}" for x in np.mean(fr, axis=0)))
    print("  a 25% column near 1.00 means extra budget bought almost nothing.")


def anytime_curve(raw, arms, base, unit, checkpoints=None):
    tr = _traces(raw)
    if not tr:
        return
    budget = min(r["trace"][-1][0] for r in tr)
    max_setup = max(int(_pick(r, SETUP_KEYS, 0) or 0) for r in tr)
    if checkpoints is None:
        lo = max(1000, max_setup)
        checkpoints = sorted({int(x) for x in np.geomspace(lo, budget - max_setup, 8)})
    for axis in ("total", "search"):
        print(f"\nanytime paired delta vs '{base}' on the {axis}-decode axis (percent)")
        print(f"  {'decodes':>9}" + "".join(f"{a:>14}" for a in arms if a != base))
        for c in checkpoints:
            def at(r, c=c):
                off = int(_pick(r, SETUP_KEYS, 0) or 0) if axis == "search" else 0
                if r["trace"][-1][0] < c + off:
                    return None
                return best_at(r["trace"], c, off)
            vals = unit_values(tr, at, unit)
            row = []
            for a in arms:
                if a == base:
                    continue
                d = paired_deltas(vals, a, base)
                row.append(f"{np.mean(d):>+9.3f} n={len(d):<2}" if d else f"{'-':>14}")
            print(f"  {c:>9,}" + "".join(row))
    print("  search axis: each arm is read after the same number of its own search decodes.")


def main(a):
    report(load(a.path), a.baseline, a.boot, a.seed, a.unit, a.curve,
           [int(x) for x in a.checkpoints.split(",")] if a.checkpoints else None)
