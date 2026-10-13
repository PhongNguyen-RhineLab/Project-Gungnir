"""
run_real.py
===========
Runs the five proposal-distribution arms against the REAL pretrained CVAE-Opt
decoder. This is the experiment; latent_search.py --mode selftest was only a
wiring check.

    python3 run_real.py --repo ./CVAE-Opt --size 100 --instances 5 \
        --seeds 3 --budget 180600 --device cuda

Budget note
-----------
180,600 is CVAE-Opt's own budget: population 600 x (300 DE iterations + the
initial population). Use it if you want the published 1.364% CVRP100 gap to be
a meaningful reference point. Anything much smaller measures pre-convergence
behaviour, which ranked the arms differently in our self-test.

Setup accounting
----------------
The pool that trains the score net and the flow is built THROUGH the budgeted
decoder, so those decodes are charged to the arms that use them. Pass
--amortize N if in your intended deployment the prior is trained once offline
and reused across N instances; then it is charged 1/N.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import cvae_opt_adapter as A
import latent_search as L

MODELS = {
    (100, "CVRP"): "cvrp_100_model_78580.pt",
    (50, "CVRP"): "cvrp_50_model_96631.pt",
    (20, "CVRP"): "cvrp_20_model_30047.pt",
}
TESTSET = {100: "vrp100_test_small_seed1235.pkl",
           50: "vrp50_test_small_seed1235.pkl",
           20: "vrp20_test_small_seed1235.pkl"}

# LKH3 reference costs on these 1000-instance sets, from the CVAE-Opt paper.
# Used only to print a gap; the arm-vs-arm comparison does not depend on them.
LKH3 = {20: 6.14, 50: 10.38, 100: 15.54}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="./CVAE-Opt")
    p.add_argument("--size", type=int, default=100)
    p.add_argument("--instances", type=int, default=5)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--budget", type=int, default=180600)
    p.add_argument("--pop", type=int, default=600)
    p.add_argument("--pool", type=int, default=6000, help="decodes spent fitting priors")
    p.add_argument("--amortize", type=int, default=1)
    p.add_argument("--lam", type=float, default=5.0)
    p.add_argument("--chains", type=int, default=0, help="0 = budget//1500")
    p.add_argument("--device", default="auto",
                   help="auto | cuda | cuda:0 | mps | cpu")
    p.add_argument("--batch", type=int, default=600)
    p.add_argument("--bench", action="store_true", help="sweep batch size, then exit")
    p.add_argument("--landscape", action="store_true", help="Stage 0 only, then exit")
    p.add_argument("--out", default="real_results.json")
    p.add_argument("--arms", default="de,flow_de,langevin_de",
                   help="comma-separated subset of arms to run")
    p.add_argument("--start", type=int, default=0,
                   help="first instance index; use to get fresh instances")
    p.add_argument("--flow-epochs", type=int, default=800)
    p.add_argument("--score-epochs", type=int, default=400)
    p.add_argument("--keep-traces", action="store_true")
    a = p.parse_args()

    dev = L.set_device(a.device)
    repo = A._preflight(a.repo)
    mp = os.path.join(repo, "models", MODELS[(a.size, "CVRP")])
    model, cfg, z_bound, meta = A.load_model(repo, mp, str(dev), a.batch)
    insts = A.read_cvrp_instances(
        os.path.join(repo, "instances", "cvrp", "test", TESTSET[a.size]), a.size)
    print(f"CVRP{a.size} | dz={cfg.search_space_size} Z_bound={z_bound:.4f} "
          f"| paper avg_gap={meta['avg_gap']:.4f}%")
    print(f"device: {L.device_report(dev)}")
    print(f"repo:   {repo}")
    print(f"budget={a.budget} pool={a.pool} (amortized over {a.amortize}) "
          f"pop={a.pop}\n")

    if a.bench:
        d = A.CVAEOptDecoder(model, cfg, insts[0], dev, a.batch)
        for bs in (128, 300, 600, 1200):
            d.batch_size = bs
            print(f"  batch={bs:<6} {d.benchmark(bs):8.1f} decodes/sec"
                  f"   -> {a.budget/max(d.benchmark(bs),1e-9)/60:6.1f} min per arm per instance")
        return

    if a.landscape:
        d = A.CVAEOptDecoder(model, cfg, insts[0], dev, a.batch)
        L.print_landscape(d, np.random.default_rng(0), n_ref=1024, per_shell=256,
                          radii=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0), box=d.box(z_bound))
        return

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    chains = a.chains or max(8, a.budget // 1500)
    results = []

    for i in range(a.start, a.start + a.instances):
        dec = A.CVAEOptDecoder(model, cfg, insts[i], dev, a.batch)
        box = dec.box(z_bound)
        rng = np.random.default_rng(1000 + i)

        # Prior-training pool, drawn from the same box every arm searches in.
        pd = L.BudgetedDecoder(dec, a.pool)
        lat, qual = L.build_pool(pd, rng, a.pool, box=box)
        scorenet = L.train_scorenet(lat, qual, epochs=a.score_epochs, seed=i)
        flow = L.train_flow(lat, epochs=a.flow_epochs, seed=i)

        cfgs = {
            "random":      dict(pop=a.pop, box=box),
            "de":          dict(pop=a.pop, box=box),
            "flow_de":     dict(pop=a.pop, box=box, flow=flow),
            "langevin_de": dict(pop=a.pop, box=box, scorenet=scorenet),
            "metropolis":  dict(chains=chains, lam=a.lam, sigma_prop=0.25, box=box),
        }
        for s in range(a.seeds):
            for arm in arms:
                su = a.pool if arm in L.PRIOR_ARMS else 0
                r = L.run_arm(arm, dec, a.budget, seed=s * 97 + i,
                              setup_evals=su, amortize_over=a.amortize, **cfgs[arm])
                r["instance"] = i
                results.append(r)
        line = "  ".join(
            f"{x}={min(r['best'] for r in results if r['instance'] == i and r['arm'] == x):.3f}"
            for x in arms)
        print(f"  instance {i}: {line}")

    print("\n" + L.summarise(results, arms))
    ref = LKH3.get(a.size)
    if ref:
        print(f"\nLKH3 reference for CVRP{a.size} (full 1000-instance set): {ref}")
        print("  gaps below are vs that reference; note your subset is smaller")
        for arm in arms:
            v = [r["best"] for r in results if r["arm"] == arm]
            print(f"    {arm:<14}{np.mean(v):>9.4f}{100*(np.mean(v)/ref-1):>9.3f}%")
    ar = [r["diagnostics"].get("accept_rate") for r in results if r["arm"] == "metropolis"]
    ar = [x for x in ar if x is not None]
    if ar:
        print(f"\nmetropolis acceptance {np.mean(ar):.3f} (chains={chains}, "
              f"iters~{a.budget // chains}); retune if outside 0.2-0.5")

    with open(a.out, "w") as f:
        drop = () if a.keep_traces else ("trace",)
        json.dump([{k: v for k, v in r.items() if k not in drop} for r in results],
                  f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
