"""The real-decoder experiment (formerly run_real.py).

Pool modes
----------
per-instance (default, same as before)
    The prior for instance i is fit on a pool decoded on instance i itself and the
    full pool is charged to that instance's prior arms. With --amortize N > 1 this
    mode UNDER-charges: the prior has still seen this instance, so N-fold sharing
    is accounting only. A warning is printed.

shared
    The prior is fit ONCE on held-out instances (--pool-start, --pool-instances),
    frozen, and reused on every evaluation instance. The pool cost is divided by
    --amortize (default: the number of evaluation instances). This is the honest
    transfer/amortization test.

Results are written after every instance, so an interrupted run keeps its data.
"""

from __future__ import annotations

import numpy as np

from . import cvae_opt as C
from .budget import BudgetedDecoder
from .device import device_report, set_device
from .experiment import (Priors, acceptance_note, arm_configs, build_pool, dump_results,
                         instance_line, print_landscape, run_arm, setup_for, summarise)
from .samplers import PRIOR_ARMS, parse_arms


def _load(a):
    dev = set_device(a.device)
    repo = C.preflight(a.repo)
    model, cfg, z_bound, meta = C.load_model(repo, a.size, dev, a.batch)
    insts = C.test_instances(repo, a.size)
    print(f"CVRP{a.size} | dz={cfg.search_space_size} Z_bound={z_bound:.4f} "
          f"| checkpoint avg_gap={meta['avg_gap']:.4f}%")
    print(f"device: {device_report(dev)}")
    print(f"repo:   {repo}")
    return dev, model, cfg, z_bound, insts


def bench(a):
    dev, model, cfg, z_bound, insts = _load(a)
    d = C.CVAEOptDecoder(model, cfg, insts[0], dev, a.batch)
    for bs in (128, 300, 600, 1200):
        d.batch_size = bs
        rate = d.benchmark(bs)
        print(f"  batch={bs:<6} {rate:8.1f} decodes/sec   -> "
              f"{a.budget / max(rate, 1e-9) / 60:6.1f} min per arm per instance")


def landscape(a):
    dev, model, cfg, z_bound, insts = _load(a)
    d = C.CVAEOptDecoder(model, cfg, insts[a.start], dev, a.batch)
    print_landscape(d, np.random.default_rng(0), n_ref=1024, per_shell=256,
                    radii=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0), box=d.box(z_bound))


def _instance_pool(model, cfg, insts, idx, dev, batch, size, z_bound):
    dec = C.CVAEOptDecoder(model, cfg, insts[idx], dev, batch)
    rng = np.random.default_rng(1000 + idx)
    return build_pool(BudgetedDecoder(dec, size), rng, size, box=dec.box(z_bound))


def _shared_priors(a, arms, model, cfg, insts, dev, z_bound, eval_range):
    pool_range = range(a.pool_start, a.pool_start + a.pool_instances)
    overlap = set(pool_range) & set(eval_range)
    if overlap:
        raise SystemExit(f"--pool-mode shared: pool instances overlap evaluation "
                         f"instances {sorted(overlap)[:5]}... choose a disjoint --pool-start")
    per = a.pool // len(pool_range)
    lats, quals = [], []
    for j in pool_range:
        lat, qual = _instance_pool(model, cfg, insts, j, dev, a.batch, per, z_bound)
        lats.append(lat)
        # costs differ in scale across instances; standardise before pooling
        quals.append((qual - qual.mean()) / (qual.std() + 1e-8))
    print(f"shared prior: {per} decodes x {len(pool_range)} held-out instances "
          f"[{pool_range.start}, {pool_range.stop})")
    return Priors.fit(np.concatenate(lats), np.concatenate(quals), arms,
                      a.score_epochs, a.flow_epochs, seed=a.pool_start,
                      pool_evals=per * len(pool_range))


def run(a):
    arms = parse_arms(a.arms)
    dev, model, cfg, z_bound, insts = _load(a)
    eval_range = range(a.start, a.start + a.instances)
    chains = a.chains or max(8, a.budget // 1500)
    uses_prior = bool(PRIOR_ARMS & set(arms))

    amortize = a.amortize
    shared = None
    if uses_prior and a.pool_mode == "shared":
        shared = _shared_priors(a, arms, model, cfg, insts, dev, z_bound, eval_range)
        amortize = a.amortize if a.amortize > 1 else a.instances
    elif uses_prior and a.amortize > 1:
        print(f"WARNING: --amortize {a.amortize} with --pool-mode per-instance charges "
              f"{a.pool // a.amortize} decodes for a prior fit on {a.pool} decodes of the "
              f"SAME instance. That is not amortization; use --pool-mode shared.")
    print(f"budget={a.budget} pool={a.pool} mode={a.pool_mode} amortize={amortize} "
          f"pop={a.pop} arms={','.join(arms)}\n")

    results = []
    for i in eval_range:
        dec = C.CVAEOptDecoder(model, cfg, insts[i], dev, a.batch)
        box = dec.box(z_bound)
        priors = shared
        if uses_prior and shared is None:
            lat, qual = _instance_pool(model, cfg, insts, i, dev, a.batch, a.pool, z_bound)
            priors = Priors.fit(lat, qual, arms, a.score_epochs, a.flow_epochs, seed=i,
                                pool_evals=a.pool)
        cfgs = arm_configs(arms, a.pop, box, priors, chains=chains, lam=a.lam,
                           sigma_prop=a.sigma_prop)
        for s in range(a.seeds):
            for arm in arms:
                r = run_arm(arm, dec, a.budget, seed=s * 97 + i,
                            setup_evals=setup_for(arm, priors), amortize_over=amortize,
                            **cfgs[arm])
                r["instance"] = i
                results.append(r)
        print(instance_line(results, i, arms), flush=True)
        dump_results(results, a.out, a.keep_traces)

    print("\n" + summarise(results, arms))
    ref = C.LKH3.get(a.size)
    if ref:
        print(f"\nLKH3 mean on the full 1000-instance CVRP{a.size} set: {ref}. Display only; "
              f"a {a.instances}-instance subset mean differs from it by sampling alone.")
        for arm in arms:
            v = np.mean([r["best"] for r in results if r["arm"] == arm])
            print(f"    {arm:<14}{v:>9.4f}{100 * (v / ref - 1):>9.3f}%")
    note = acceptance_note(results)
    if note:
        print(f"\n{note} (chains={chains}, iters~{a.budget // chains})")
    print(f"\nwrote {a.out}")
