"""The five arms. Each sampler takes (BudgetedDecoder, rng, **cfg) and runs until
the budget is exhausted.

  random       z ~ N(0, I) clipped to the box, keep best         (null)
  de           DE/rand/1/bin, uniform init in the box             (CVAE-Opt baseline)
  flow_de      flow-prior init, then the same DE                  (cheap competitor)
  langevin_de  score-model annealed Langevin init, then DE        (GEN-CIM design)
  metropolis   cost-tilted Metropolis with adaptive proposal      (LGS design)

The three DE arms differ ONLY in the initial population.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from .budget import BudgetedDecoder
from .device import cpu_gen
from .priors import FlowPrior, ScoreNet, annealed_langevin


def clip(Z, box):
    return Z if box is None else np.clip(Z, box[0], box[1])


def sampler_random(d: BudgetedDecoder, rng, pop=600, box=None, **_):
    while not d.exhausted:
        d(clip(rng.standard_normal((pop, d.dz)).astype(np.float32), box))


def de_loop(d: BudgetedDecoder, rng, P, costs, CR=0.95, F=0.3, box=None):
    """DE/rand/1/bin, batched. CVAE-Opt used pop=600, CR=0.95, F=0.3."""
    pop = len(P)
    while not d.exhausted:
        a, b, c = (rng.integers(0, pop, pop) for _ in range(3))
        V = P[a] + F * (P[b] - P[c])
        cross = rng.random((pop, d.dz)) < CR
        jrand = rng.integers(0, d.dz, pop)
        cross[np.arange(pop), jrand] = True
        U = clip(np.where(cross, V, P).astype(np.float32), box)
        cu = d(U)
        better = cu < costs
        P[better], costs[better] = U[better], cu[better]
    return P, costs


def sampler_de(d: BudgetedDecoder, rng, pop=600, box=None, **cfg):
    if box is None:
        P = rng.standard_normal((pop, d.dz)).astype(np.float32)
    else:
        P = rng.uniform(box[0], box[1], (pop, d.dz)).astype(np.float32)
    de_loop(d, rng, P, d(P), box=box, **cfg)


def sampler_langevin_de(d: BudgetedDecoder, rng, scorenet: ScoreNet, pop=600,
                        lan_steps=60, eta=0.01, box=None, **cfg):
    P = annealed_langevin(scorenet, pop, d.dz, steps=lan_steps, eta=eta,
                          rng_seed=int(rng.integers(1 << 30)), box=box)
    P = clip(P.astype(np.float32), box)
    de_loop(d, rng, P, d(P), box=box, **cfg)


def sampler_flow_de(d: BudgetedDecoder, rng, flow: FlowPrior, pop=600, box=None, **cfg):
    g = cpu_gen(int(rng.integers(1 << 30)))
    P = clip(flow.sample(pop, g).to("cpu").numpy().astype(np.float32), box)
    de_loop(d, rng, P, d(P), box=box, **cfg)


def sampler_metropolis(d: BudgetedDecoder, rng, chains=600, lam=5.0, sigma_prop=0.1,
                       gamma=0.379, log_prior=None, box=None, adapt=True,
                       target_accept=0.3, adapt_every=10, **_):
    """Cost-tilted Metropolis.

    Target   pi(z) ~ prior(z) exp(-lam * cost(decode(z)))
    Proposal z' = z + gamma (z_i1 - z_i2) + sigma_prop eps   (symmetric)
    Scales   Robbins-Monro on log sigma_prop and log gamma toward target_accept.

    At fixed budget, iterations = budget / chains. A chain needs iterations, not
    width: on the self-test (12k budget) chains 200/50/20/8 gave 21.41/20.31/
    19.83/18.56 against DE 16.21. Never set chains = DE population. Use `tune`.
    """
    if log_prior is None:
        log_prior = lambda Z: -0.5 * (Z.astype(np.float64) ** 2).sum(-1)  # noqa: E731

    Z = clip(rng.standard_normal((chains, d.dz)).astype(np.float32), box)
    C = d(Z)
    lp = log_prior(Z)
    n_acc = n_prop = 0
    log_s, log_g = math.log(sigma_prop), math.log(max(gamma, 1e-6))
    win_acc, win_n, it = 0, 0, 0

    while not d.exhausted:
        i1, i2 = rng.integers(0, chains, chains), rng.integers(0, chains, chains)
        Zp = clip((Z + math.exp(log_g) * (Z[i1] - Z[i2])
                   + math.exp(log_s) * rng.standard_normal((chains, d.dz))
                   ).astype(np.float32), box)
        Cp = d(Zp)
        lpp = log_prior(Zp)
        log_alpha = -lam * (Cp - C) + (lpp - lp)
        acc = np.log(rng.random(chains) + 1e-300) < log_alpha
        acc &= np.isfinite(Cp)
        Z[acc], C[acc], lp[acc] = Zp[acc], Cp[acc], lpp[acc]
        k = int(np.isfinite(Cp).sum())
        n_acc += int(acc.sum())
        n_prop += k
        win_acc += int(acc.sum())
        win_n += k
        it += 1
        if adapt and it % adapt_every == 0 and win_n > 0:
            rate = 0.5 / math.sqrt(1 + it / adapt_every)
            delta = rate * ((win_acc / win_n) - target_accept)
            log_s = float(np.clip(log_s + delta, math.log(1e-4), math.log(5.0)))
            log_g = float(np.clip(log_g + delta, math.log(1e-4), math.log(1.0)))
            win_acc = win_n = 0
    # accept ~0: chain stuck (lam too cold / proposal too wide)
    # accept ~1: random search; 0.2-0.5 and still losing: no local structure
    d.diagnostics = {"accept_rate": n_acc / max(1, n_prop),
                     "final_sigma_prop": math.exp(log_s),
                     "final_gamma": math.exp(log_g)}


# --------------------------------------------------------------------- registry

@dataclass(frozen=True)
class Arm:
    name: str
    fn: Callable
    prior: Optional[str] = None     # "flow" | "scorenet" | None

    @property
    def needs_prior(self) -> bool:
        return self.prior is not None


ARMS: Dict[str, Arm] = {a.name: a for a in (
    Arm("random", sampler_random),
    Arm("de", sampler_de),
    Arm("flow_de", sampler_flow_de, prior="flow"),
    Arm("langevin_de", sampler_langevin_de, prior="scorenet"),
    Arm("metropolis", sampler_metropolis),
)}

PRIOR_ARMS = frozenset(n for n, a in ARMS.items() if a.needs_prior)
ALL_ARMS = list(ARMS)


def parse_arms(spec: str):
    arms = [x.strip() for x in spec.split(",") if x.strip()]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise ValueError(f"unknown arm(s) {bad}; choose from {ALL_ARMS}")
    return arms
