"""
Calibration harness for random-keys CVRP decoders (numba-accelerated).

Reproduces the Opt-DE row of CVAE-Opt (Hottung et al., ICLR 2021, Table 3)
to give an independent yardstick for a random-keys `de` arm.

  bean  - Bean (1994) sequential fill. What Opt-DE implements.
  split - Prins (2004) optimal split of the giant tour. What BRKGA does.

Instances follow Kool et al. (2019) CVRP100.
"""
import numpy as np
import time
import argparse
from numba import njit


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
    cost += D[prev, 0]
    return cost


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
def _eval_pop(P, D, dem, Q, mode):
    m = P.shape[0]
    out = np.empty(m)
    for i in range(m):
        order = np.argsort(P[i])
        if mode == 0:
            out[i] = _bean(order, D, dem, Q)
        else:
            out[i] = _split(order, D, dem, Q)
    return out


@njit(cache=True)
def _de(D, dem, Q, mode, n, pop_size, iters, seed, F, CR):
    np.random.seed(seed)
    P = np.random.random((pop_size, n))
    fit = _eval_pop(P, D, dem, Q, mode)
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
                    if v < 0.0:
                        v = 0.0
                    elif v > 1.0:
                        v = 1.0
                    trial[i, j] = v
                else:
                    trial[i, j] = P[i, j]
        tfit = _eval_pop(trial, D, dem, Q, mode)
        for i in range(pop_size):
            if tfit[i] < fit[i]:
                fit[i] = tfit[i]
                for j in range(n):
                    P[i, j] = trial[i, j]
    return fit.min()


MODES = {"bean": 0, "split": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--pop", type=int, default=600)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--decoders", type=str, default="bean,split")
    args = ap.parse_args()

    decs = args.decoders.split(",")
    res = {d: [] for d in decs}

    t_start = time.time()
    for k in range(args.instances):
        D, dem, Q = make_instance(args.n, seed=1000 + args.start + k)
        line = f"inst {args.start + k:3d}:"
        for d in decs:
            c = _de(D, dem, Q, MODES[d], args.n, args.pop, args.iters, 7, 0.3, 0.95)
            res[d].append(c)
            line += f"  {d}={c:7.3f}"
        print(line, flush=True)

    budget = args.pop * (args.iters + 1)
    print()
    print(f"decode budget per run: {budget:,}   total wall {time.time()-t_start:.0f}s")
    print(f"{'decoder':10s} {'mean cost':>10s}")
    print("-" * 22)
    for d in decs:
        print(f"{d:10s} {np.mean(res[d]):10.3f}")
    print()
    print("published CVRP100 reference (1000 instances):")
    print("  LKH3 15.54 | NLNS 15.54 | CVAE-Opt-DE 15.75 | CVAE-Opt-RS 16.11 | Opt-DE 23.97")


if __name__ == "__main__":
    main()
