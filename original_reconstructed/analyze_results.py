"""
analyze_results.py
==================
Reads the JSON written by run_real.py (or latent_search.py --out) and answers
one question: are the arms actually indistinguishable, or is the experiment
just underpowered?

    python3 analyze_results.py results.json
    python3 analyze_results.py results.json --baseline de --boot 20000

Why paired deltas
-----------------
Raw mean cost across instances is dominated by instance-to-instance variation,
which is often 10x the arm-to-arm effect. Example: instance A costs 15.2 and
instance B costs 21.8 no matter which arm you use. Averaging those hides a
consistent 0.3% arm effect. Comparing arm-vs-baseline WITHIN each
(instance, seed) cell cancels that out.

Reading the output
------------------
  CI excludes zero        -> real difference
  CI tight around zero    -> real null, the arms genuinely do not separate
  CI wide (say +/- 1.5%)  -> you have no measurement yet, add instances/seeds
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------- loading

COST_KEYS = ("best", "best_cost", "cost", "obj", "objective")
INST_KEYS = ("instance", "inst", "instance_id", "inst_id")
SEED_KEYS = ("seed", "run_seed")
SETUP_KEYS = ("setup", "setup_evals", "setup_cost")
EVAL_KEYS = ("evals", "n_evals", "budget_used", "budget")


def _pick(rec, keys, default=None):
    for k in keys:
        if k in rec:
            return rec[k]
    return default


def load(path):
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        # some runs wrap the list; find the first list of dicts
        for v in raw.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                raw = v
                break
    if not isinstance(raw, list):
        sys.exit("could not find a list of result records in that file")
    return raw


def describe(raw):
    keys = sorted({k for r in raw for k in r})
    print(f"records: {len(raw)}")
    print(f"keys:    {', '.join(keys)}")
    if "trace" in keys:
        print("note:    traces present, saturation check available")
    else:
        print("note:    no traces in file, cannot check whether the search "
              "plateaued early")
    print()


# ---------------------------------------------------------------- stats

def cells(raw):
    """(instance, seed, arm) -> cost. Averages duplicates."""
    acc = defaultdict(list)
    for r in raw:
        arm = r.get("arm")
        cost = _pick(r, COST_KEYS)
        if arm is None or cost is None:
            continue
        inst = _pick(r, INST_KEYS, 0)
        seed = _pick(r, SEED_KEYS, 0)
        acc[(inst, seed, arm)].append(float(cost))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def paired_bootstrap(deltas, n_boot, rng):
    """CI on the mean of paired deltas. Resamples the pairs, not the arms."""
    d = np.asarray(deltas, dtype=float)
    if len(d) < 2:
        return float(np.mean(d)) if len(d) else float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_test_p(deltas):
    """Two-sided sign test, exact binomial. Ties dropped."""
    d = [x for x in deltas if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0, 0, 0
    wins = sum(1 for x in d if x < 0)          # arm cheaper than baseline
    k = min(wins, n - wins)
    # sum of binomial tail, both sides
    from math import comb
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(p, 1.0), wins, n


def verdict(lo, hi):
    width = hi - lo
    if np.isnan(width):
        return "too few pairs"
    if lo > 0 or hi < 0:
        return "DIFFERENT"
    if width < 0.5:
        return "null (tight)"
    if width < 1.5:
        return "null (loose)"
    return "UNDERPOWERED"


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--baseline", default="de",
                    help="arm to compare everything against (default: de)")
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    raw = load(a.path)
    describe(raw)

    c = cells(raw)
    arms = sorted({k[2] for k in c})
    pairs = sorted({(k[0], k[1]) for k in c})

    if not arms:
        sys.exit("no 'arm' field found in the records")

    base = a.baseline if a.baseline in arms else arms[0]
    if base != a.baseline:
        print(f"baseline '{a.baseline}' not present, using '{base}' instead\n")

    # ---- raw means, the misleading view, printed for comparison
    print("raw mean cost per arm (instance variation NOT removed)")
    print(f"  {'arm':<14}{'mean':>10}{'std':>10}{'n':>6}")
    for arm in arms:
        v = [c[k] for k in c if k[2] == arm]
        print(f"  {arm:<14}{np.mean(v):>10.4f}{np.std(v):>10.4f}{len(v):>6}")
    print()

    # ---- paired view, the one that matters
    rng = np.random.default_rng(a.seed)
    print(f"paired delta vs '{base}', in percent. negative = arm is better")
    print(f"  {'arm':<14}{'mean%':>9}{'CI low':>9}{'CI high':>9}"
          f"{'wins':>7}{'p':>8}   verdict")
    print("  " + "-" * 68)

    complete = 0
    for arm in arms:
        if arm == base:
            continue
        deltas = []
        for p in pairs:
            kb, ka = (p[0], p[1], base), (p[0], p[1], arm)
            if kb in c and ka in c and c[kb] > 0:
                deltas.append(100.0 * (c[ka] / c[kb] - 1.0))
        complete = max(complete, len(deltas))
        if not deltas:
            print(f"  {arm:<14}{'no paired cells':>40}")
            continue
        m, lo, hi = paired_bootstrap(deltas, a.boot, rng)
        p, wins, n = sign_test_p(deltas)
        print(f"  {arm:<14}{m:>9.3f}{lo:>9.3f}{hi:>9.3f}"
              f"{wins:>4}/{n:<3}{p:>7.3f}   {verdict(lo, hi)}")

    print(f"\n  {complete} paired cells "
          f"({len({p[0] for p in pairs})} instances x "
          f"{len({p[1] for p in pairs})} seeds)")
    if complete < 15:
        print("  warning: fewer than 15 pairs. the bootstrap CI is not "
              "trustworthy at this sample size, and a 'null' result here "
              "means 'no measurement', not 'no effect'.")

    # ---- budget accounting sanity
    setups = defaultdict(set)
    for r in raw:
        s = _pick(r, SETUP_KEYS)
        if s is not None:
            setups[r.get("arm")].add(int(s))
    if setups:
        print("\nsetup evals charged per arm")
        for arm in arms:
            print(f"  {arm:<14}{sorted(setups.get(arm, {0}))}")
        prior = [x for x in ("flow_de", "langevin_de") if x in arms]
        bad = [x for x in prior if setups.get(x) in (None, {0})]
        if bad:
            print(f"  WARNING: prior arms {bad} were charged zero setup cost. "
                  "their numbers are not comparable to the others.")

    # ---- metropolis health
    ar = [r["diagnostics"]["accept_rate"] for r in raw
          if isinstance(r.get("diagnostics"), dict)
          and "accept_rate" in r["diagnostics"]]
    if ar:
        m = float(np.mean(ar))
        flag = "" if 0.15 <= m <= 0.6 else "   <-- outside 0.15-0.6, retune"
        print(f"\nmetropolis mean acceptance: {m:.3f}{flag}")

    # ---- saturation, only if traces survived
    tr = [r for r in raw if isinstance(r.get("trace"), list) and len(r["trace"]) > 4]
    if tr:
        print("\nsaturation: fraction of total improvement reached by each "
              "quarter of the budget")
        for arm in arms:
            ts = [r["trace"] for r in tr if r.get("arm") == arm]
            if not ts:
                continue
            fr = []
            for t in ts:
                y = np.minimum.accumulate(np.asarray(t, dtype=float))
                span = y[0] - y[-1]
                if span <= 0:
                    continue
                fr.append([(y[0] - y[int(q * (len(y) - 1))]) / span
                           for q in (0.25, 0.5, 0.75, 1.0)])
            if fr:
                v = np.mean(fr, axis=0)
                print(f"  {arm:<14}" + "".join(f"{x:>8.2f}" for x in v))
        print("  if the 25% column is already near 1.00, extra budget bought "
              "nothing and you can iterate at a much smaller budget.")


if __name__ == "__main__":
    main()
