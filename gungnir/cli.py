"""Single entry point:  python -m gungnir <command> [options]

  device     list cpu / cuda / mps and smoke-test the chosen one
  selftest   all arms on the random-keys surrogate (wiring check)
  landscape  Stage 0 smoothness check (--real for the CVAE-Opt decoder)
  tune       Metropolis chains x lam sweep on the surrogate
  bench      decoder throughput vs batch size on the real decoder
  run        the real-decoder experiment (formerly run_real.py)
  analyze    paired statistics on a results JSON (formerly analyze_results.py)
  calib      random-keys DE yardstick (formerly calib2.py)
  sweep      random-keys budget curve (formerly sweep.py)
"""

from __future__ import annotations

import argparse

import numpy as np


def _real_args(p):
    p.add_argument("--repo", default="./CVAE-Opt")
    p.add_argument("--size", type=int, default=100, choices=(20, 50, 100))
    p.add_argument("--device", default="auto", help="auto | cuda | cuda:N | mps | cpu")
    p.add_argument("--batch", type=int, default=600)
    p.add_argument("--budget", type=int, default=180600)
    p.add_argument("--start", type=int, default=0, help="first test-instance index")


def build_parser():
    ap = argparse.ArgumentParser(prog="gungnir", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("device", help="list backends and smoke-test one")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N | mps | all")

    p = sub.add_parser("selftest")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--instances", type=int, default=5)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--budget", type=int, default=30000)
    p.add_argument("--pop", type=int, default=300)
    p.add_argument("--lam", type=float, default=5.0)
    p.add_argument("--pool", type=int, default=4000)
    p.add_argument("--arms", default="random,de,flow_de,langevin_de,metropolis")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    p.add_argument("--keep-traces", action="store_true")

    p = sub.add_parser("landscape")
    p.add_argument("--real", action="store_true", help="use the CVAE-Opt decoder")
    p.add_argument("--n", type=int, default=50, help="surrogate instance size")
    _real_args(p)

    p = sub.add_parser("tune")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--budget", type=int, default=30000)
    p.add_argument("--seeds", type=int, default=2)

    p = sub.add_parser("bench")
    _real_args(p)

    p = sub.add_parser("run")
    _real_args(p)
    p.add_argument("--instances", type=int, default=5)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--pop", type=int, default=600)
    p.add_argument("--arms", default="de,flow_de,langevin_de")
    p.add_argument("--pool", type=int, default=6000, help="decodes spent fitting priors")
    p.add_argument("--pool-mode", choices=("per-instance", "shared"), default="per-instance")
    p.add_argument("--pool-start", type=int, default=900,
                   help="shared mode: first held-out instance for the pool")
    p.add_argument("--pool-instances", type=int, default=24,
                   help="shared mode: number of held-out pool instances")
    p.add_argument("--amortize", type=int, default=1,
                   help="divide the pool charge by N (shared mode default: --instances)")
    p.add_argument("--flow-epochs", type=int, default=800)
    p.add_argument("--score-epochs", type=int, default=400)
    p.add_argument("--lam", type=float, default=5.0)
    p.add_argument("--chains", type=int, default=0, help="0 = max(8, budget // 1500)")
    p.add_argument("--sigma-prop", type=float, default=0.25)
    p.add_argument("--keep-traces", action="store_true")
    p.add_argument("--out", default="real_results.json")

    p = sub.add_parser("analyze")
    p.add_argument("path")
    p.add_argument("--baseline", default="de")
    p.add_argument("--boot", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--unit", choices=("instance", "cell"), default="instance")
    p.add_argument("--curve", action="store_true", help="anytime curve (needs traces)")
    p.add_argument("--checkpoints", default=None, help="comma-separated decode counts")

    p = sub.add_parser("calib")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--instances", type=int, default=10)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--pop", type=int, default=600)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--decoders", default="bean,split")

    p = sub.add_parser("sweep")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--instances", type=int, default=4)
    p.add_argument("--pop", type=int, default=600)
    p.add_argument("--budgets", default="60000,240000,960000")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)

    if a.cmd == "device":
        from .device import available, device_report, resolve_device, smoke_test
        info = available()
        print(f"cpu   yes\ncuda  {', '.join(info['cuda']) or 'no'}\n"
              f"mps   built={info['mps_built']} available={info['mps_available']} "
              f"fallback={'on' if info['mps_fallback'] else 'off'}")
        specs = (["cpu"] + [f"cuda:{i}" for i in range(len(info["cuda"]))]
                 + (["mps"] if info["mps_available"] else [])) if a.device == "all" else [a.device]
        for spec in specs:
            dev = resolve_device(spec)
            print(f"\n{device_report(dev)}\n  smoke test: {smoke_test(dev)}")
    elif a.cmd == "selftest":
        from .device import set_device
        from .experiment import selftest
        from .samplers import parse_arms
        set_device(a.device)
        selftest(a.n, a.instances, a.seeds, a.budget, a.pop, a.lam, a.pool,
                 parse_arms(a.arms), a.out, a.keep_traces)
    elif a.cmd == "landscape":
        if a.real:
            from .runner import landscape
            landscape(a)
        else:
            from .experiment import print_landscape
            from .problems import RandomKeysDecoder, make_instance
            print_landscape(RandomKeysDecoder(make_instance(a.n, np.random.default_rng(1000))),
                            np.random.default_rng(0))
    elif a.cmd == "tune":
        from .experiment import tune_metropolis
        from .problems import RandomKeysDecoder, make_instance
        print(f"tuning metropolis | budget={a.budget}")
        tune_metropolis(RandomKeysDecoder(make_instance(a.n, np.random.default_rng(1000))),
                        a.budget, seeds=a.seeds)
    elif a.cmd == "bench":
        from .runner import bench
        bench(a)
    elif a.cmd == "run":
        from .runner import run
        run(a)
    elif a.cmd == "analyze":
        from .analysis import main as analyze
        analyze(a)
    elif a.cmd == "calib":
        from .calibration import calib
        calib(a)
    elif a.cmd == "sweep":
        from .calibration import sweep
        sweep(a)


if __name__ == "__main__":
    main()