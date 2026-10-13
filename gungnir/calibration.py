"""Random-keys DE yardstick (formerly calib2.py and sweep.py).

Plain DE/rand/1/bin over random keys in [0,1]^n with two decoders:
  bean   Bean (1994) sequential fill (what CVAE-Opt's Opt-DE row uses)
  split  Prins (2004) optimal split of the giant tour (what BRKGA uses)

Deliberately basic: no restarts, no local search. Read the curve as a lower bound
on what random keys can do. Note make_instance here draws its own coordinates;
these are not the CVAE-Opt test instances.
"""

from __future__ import annotations

import time

import numpy as np

try:
    from numba import njit
except ImportError:                       # pure-python fallback, very slow
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

MODES = {"bean": 0, "split": 1}
REFERENCE = "LKH3 15.54 | NLNS 15.54 | CVAE-Opt-DE 15.75 | CVAE-Opt-RS 16.11 | Opt-DE 23.97"


def make_instance(n, seed, capacity=50):
    rng = np.random.default_rng(seed)
    coords = rng.random((n + 1, 2))
    dem = rng.integers(1, 10, size=n).astype(np.int64)
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    return D, dem, capacity


@njit(cache=True)
def _bean(order, D, dem, Q):
    cost = 0.0
    load = 0
    prev = 0
    for k in range(order.shape[0]):
        c = order[k]
        node = c + 1
        if load + dem[c] > Q:
            cost += D[prev, 0]
            prev = 0
            load = 0
        cost += D[prev, node]
        load += dem[c]
        prev = node
    return cost + D[prev, 0]


@njit(cache=True)
def _split(order, D, dem, Q):
    n = order.shape[0]
    INF = 1e18
    best = np.full(n + 1, INF)
    best[0] = 0.0
    for i in range(n):
        if best[i] >= INF:
            continue
        load = 0
        rc = 0.0
        prev = 0
        for j in range(i, n):
            c = order[j]
            if load + dem[c] > Q:
                break
            load += dem[c]
            node = c + 1
            rc += D[prev, node]
            prev = node
            v = best[i] + rc + D[node, 0]
            if v < best[j + 1]:
                best[j + 1] = v
    return best[n]


@njit(cache=True)
def eval_pop(P, D, dem, Q, mode):
    out = np.empty(P.shape[0])
    for i in range(P.shape[0]):
        order = np.argsort(P[i])
        out[i] = _bean(order, D, dem, Q) if mode == 0 else _split(order, D, dem, Q)
    return out


@njit(cache=True)
def de(D, dem, Q, mode, n, pop_size, iters, seed, F, CR):
    np.random.seed(seed)
    P = np.random.random((pop_size, n))
    fit = eval_pop(P, D, dem, Q, mode)
    trial = np.empty((pop_size, n))
    for _ in range(iters):
        for i in range(pop_size):
            a = np.random.randint(0, pop_size)
            b = np.random.randint(0, pop_size)
            c = np.random.randint(0, pop_size)
            jr = np.random.randint(0, n)
            for j in range(n):
                if np.random.random() < CR or j == jr:
                    v = P[a, j] + F * (P[b, j] - P[c, j])
                    trial[i, j] = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
                else:
                    trial[i, j] = P[i, j]
        tfit = eval_pop(trial, D, dem, Q, mode)
        for i in range(pop_size):
            if tfit[i] < fit[i]:
                fit[i] = tfit[i]
                for j in range(n):
                    P[i, j] = trial[i, j]
    return fit.min()


def calib(a):
    decs = a.decoders.split(",")
    res = {d: [] for d in decs}
    t0 = time.time()
    for k in range(a.instances):
        D, dem, Q = make_instance(a.n, seed=1000 + a.start + k)
        line = f"inst {a.start + k:3d}:"
        for d in decs:
            c = de(D, dem, Q, MODES[d], a.n, a.pop, a.iters, 7, 0.3, 0.95)
            res[d].append(c)
            line += f"  {d}={c:7.3f}"
        print(line, flush=True)
    print(f"\ndecode budget per run: {a.pop * (a.iters + 1):,}   total wall {time.time() - t0:.0f}s")
    print(f"{'decoder':10s} {'mean cost':>10s}\n" + "-" * 22)
    for d in decs:
        print(f"{d:10s} {np.mean(res[d]):10.3f}")
    print(f"\npublished CVRP100 reference (1000 instances):\n  {REFERENCE}")


def sweep(a):
    budgets = [int(x) for x in a.budgets.split(",")]
    insts = [make_instance(a.n, 1000 + k) for k in range(a.instances)]
    rs = {}
    for name, mode in MODES.items():
        vals = []
        for D, dem, Q in insts:
            rng = np.random.default_rng(3)
            vals.append(min(eval_pop(rng.random((10000, a.n)), D, dem, Q, mode).min()
                            for _ in range(20)))
        rs[name] = np.mean(vals)
    print(f"{'budget':>10s} {'bean':>9s} {'split':>9s}\n" + "-" * 31)
    for B in budgets:
        row = [np.mean([de(D, dem, Q, mode, a.n, a.pop, B // a.pop, 7, 0.3, 0.95)
                        for D, dem, Q in insts]) for mode in MODES.values()]
        print(f"{B:>10,} {row[0]:9.3f} {row[1]:9.3f}", flush=True)
    print("-" * 31 + f"\n{'rand 200k':>10s} {rs['bean']:9.3f} {rs['split']:9.3f}")
    print(f"\npublished CVRP100: {REFERENCE}")
