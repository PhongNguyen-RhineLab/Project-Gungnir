"""
latent_search.py
================
Harness for comparing proposal distributions over a *frozen* latent-to-solution
decoder on CVRP.

The experimental question this is built to answer:

    Given a fixed decoder p(solution | z, instance), does a learned prior
    (score model + Langevin) or a cost-tilted target (Metropolis) find better
    CVRP solutions than N(0, I) or differential evolution, AT EQUAL DECODE
    BUDGET?

Everything is measured in *decoder calls*, never wall clock, because the arms
have wildly different per-call overheads and wall clock would hide the effect.

Arms implemented
----------------
  random      sample z ~ N(0, I), decode, keep best                (the null)
  de          differential evolution in z                          (CVAE-Opt baseline)
  langevin_de annealed Langevin warm-start with a score net, then DE  (GEN-CIM design)
  metropolis  cost-tilted Metropolis on z                          (LGS design)
  flow_de     normalizing-flow prior for init, then DE             (cheap competitor)

Decoder plug-in
---------------
Any object with `costs = decode(Z)` where Z is [B, dz] float32 works. Two are
provided: RandomKeysDecoder (self-contained, for validating this harness today)
and CVAEOptDecoder (adapter stub for the real pretrained model).

READ THIS BEFORE INTERPRETING RESULTS
-------------------------------------
RandomKeysDecoder is NOT a substitute for the learned decoder. Its latent space
is a permutation-encoding, not a learned manifold of good solutions. Numbers on
it validate that the samplers are implemented correctly; they do NOT predict
what happens on the CVAE latent. In CVAE-Opt's own ablation a handcrafted
random-keys decoder scored 23.97 on CVRP100 where the learned decoder scored
15.75. Use it to debug, then swap in the real thing.
"""

from __future__ import annotations

import argparse
import os
import json
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 0. Device handling (cuda / mps / cpu)
# ---------------------------------------------------------------------------

def resolve_device(spec: str = "auto") -> torch.device:
    """'auto' picks cuda, then mps, then cpu. Anything else is taken literally
    but validated, so a typo fails loudly instead of silently running on CPU
    for six hours."""
    spec = (spec or "auto").lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but torch.cuda.is_available() is False")
    if spec == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError(
                "mps requested but unavailable. Needs Apple Silicon and a torch "
                "build with MPS. Check torch.backends.mps.is_built().")
        # Some ops in the CVAE-Opt decoder have no MPS kernel; without this the
        # run dies partway with NotImplementedError instead of falling back.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return torch.device(spec)


DEVICE = resolve_device("auto")


def set_device(spec: str = "auto") -> torch.device:
    """Set the device used by every model in this module. Call once at startup,
    before building any ScoreNet or FlowPrior."""
    global DEVICE
    DEVICE = resolve_device(spec)
    return DEVICE


def _cpu_gen(seed: int) -> torch.Generator:
    """Generators live on CPU regardless of compute device."""
    return torch.Generator(device="cpu").manual_seed(int(seed))


def _randn(shape, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.randn(*shape, generator=gen, device="cpu").to(device)


def device_report(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda: {torch.cuda.get_device_name(0)}"
    if device.type == "mps":
        return "mps: Apple Silicon (PYTORCH_ENABLE_MPS_FALLBACK=" \
               f"{os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'unset')})"
    return f"cpu: {os.cpu_count()} cores"


# ---------------------------------------------------------------------------
# 1. Problem instances (matching the Kool et al. / CVAE-Opt generator)
# ---------------------------------------------------------------------------

# capacity by customer count, as used throughout this literature
CAPACITY = {20: 30.0, 50: 40.0, 100: 50.0, 125: 55.0, 150: 60.0}


@dataclass
class CVRPInstance:
    coords: np.ndarray      # [n+1, 2], index 0 is the depot
    demands: np.ndarray     # [n+1],    demands[0] == 0
    capacity: float
    dist: np.ndarray = field(init=False)  # [n+1, n+1]

    def __post_init__(self):
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        self.dist = np.sqrt((diff ** 2).sum(-1))

    @property
    def n(self) -> int:
        return len(self.coords) - 1


def make_instance(n: int, rng: np.random.Generator) -> CVRPInstance:
    coords = rng.random((n + 1, 2))
    demands = np.concatenate([[0.0], rng.integers(1, 10, size=n).astype(float)])
    return CVRPInstance(coords, demands, CAPACITY.get(n, 50.0))


# ---------------------------------------------------------------------------
# 2. Budgeted decoder interface
# ---------------------------------------------------------------------------

class Decoder(Protocol):
    dz: int
    def decode(self, Z: np.ndarray) -> np.ndarray: ...


class BudgetedDecoder:
    """Wraps a decoder, hard-caps total decodes, and records the anytime curve.

    Every sampler must go through this object. Batches are clipped when they
    would exceed the budget, so no arm can quietly overspend.
    """

    def __init__(self, decoder: Decoder, budget: int):
        self.decoder = decoder
        self.budget = budget
        self.used = 0
        self.setup_charged = 0
        self.best = math.inf
        self.best_z: Optional[np.ndarray] = None
        self.trace: List[tuple] = []   # (evals_used, best_so_far)
        self.diagnostics: Dict[str, float] = {}

    def charge_setup(self, evals: int, amortize_over: int = 1):
        """Charge the decodes spent building an arm's prior against its budget.

        Arms that fit a score net or a flow first must pay for the solution
        pool those models were trained on. Forgetting this is not a rounding
        error: on the self-test, a 4000-decode pool charged to nobody made the
        flow arm look 5% better than DE at a 20k budget, and giving DE the same
        4000 decodes reversed the ranking outright.

        Set `amortize_over` to the number of instances the prior is reused
        across. amortize_over=1 means the prior was fit on THIS instance and
        pays full price. A prior trained offline on 100k instances and applied
        unchanged is effectively free, and you should say so in the paper.
        """
        k = int(math.ceil(max(0, evals) / max(1, amortize_over)))
        if k == 0:
            return
        self.used += k
        self.setup_charged += k
        self.trace.append((self.used, self.best))

    @property
    def dz(self) -> int:
        return self.decoder.dz

    @property
    def exhausted(self) -> bool:
        return self.used >= self.budget

    def remaining(self) -> int:
        return max(0, self.budget - self.used)

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float32)
        k = min(len(Z), self.remaining())
        if k == 0:
            return np.full(len(Z), math.inf, dtype=np.float64)
        costs = np.asarray(self.decoder.decode(Z[:k]), dtype=np.float64)
        self.used += k
        i = int(np.argmin(costs))
        if costs[i] < self.best:
            self.best = float(costs[i])
            self.best_z = Z[i].copy()
        self.trace.append((self.used, self.best))
        if k < len(Z):
            costs = np.concatenate([costs, np.full(len(Z) - k, math.inf)])
        return costs


# ---------------------------------------------------------------------------
# 3. Reference decoder: random keys (Bean, 1994) -- for validating the harness
# ---------------------------------------------------------------------------

class RandomKeysDecoder:
    """z in R^n defines a customer permutation by argsort; split into routes
    greedily whenever capacity would be exceeded. Always feasible."""

    def __init__(self, inst: CVRPInstance):
        self.inst = inst
        self.dz = inst.n

    def decode(self, Z: np.ndarray) -> np.ndarray:
        inst, D = self.inst, self.inst.dist
        B, n = Z.shape
        order = np.argsort(Z, axis=1) + 1            # [B, n] node ids 1..n
        cost = np.zeros(B)
        load = np.zeros(B)
        cur = np.zeros(B, dtype=np.int64)            # start at depot
        for t in range(n):
            nxt = order[:, t]
            d = inst.demands[nxt]
            over = (load + d) > inst.capacity + 1e-9
            cost += np.where(over, D[cur, 0], 0.0)   # return to depot first
            cur = np.where(over, 0, cur)
            load = np.where(over, 0.0, load)
            cost += D[cur, nxt]
            load += d
            cur = nxt
        cost += D[cur, 0]
        return cost


# ---------------------------------------------------------------------------
# 4. Adapter for the real pretrained CVAE-Opt decoder
# ---------------------------------------------------------------------------

class CVAEOptDecoder:
    """Adapter for github.com/ahottung/CVAE-Opt pretrained models.

    Fill in the two TODOs. Notes that matter:

    * Decode GREEDILY (argmax at each step). CVAE-Opt's reported numbers use
      greedy decoding; switching to sampling silently changes the comparison.
    * Keep the model in eval() / no_grad().
    * Batch it. The whole experiment is decoder-forward-pass bound.
    * The latent search box is NOT N(0, I). CVAE-Opt bounds the box to cover
      99% of the latents obtained by encoding held-out validation instances,
      because the encoder posterior drifts from the prior. Compute those bounds
      once (see `fit_latent_box`) and pass them to every sampler, or Langevin
      will wander into regions the decoder has never seen and you will measure
      a failure of the box, not of the sampler.
    """

    def __init__(self, model, instance_encoding, dz: int, batch_size: int = 600):
        self.model = model
        self.enc = instance_encoding
        self.dz = dz
        self.batch_size = batch_size

    @torch.no_grad()
    def decode(self, Z: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(Z), self.batch_size):
            zb = torch.as_tensor(Z[i:i + self.batch_size], device=DEVICE)
            # TODO: tours = self.model.decode_greedy(self.enc, zb)
            # TODO: out.append(tour_lengths(tours).cpu().numpy())
            raise NotImplementedError("wire in the pretrained decoder here")
        return np.concatenate(out)


def fit_latent_box(latents: np.ndarray, coverage: float = 0.99):
    """Per-dimension bounds covering `coverage` of encoded latents."""
    lo = np.quantile(latents, (1 - coverage) / 2, axis=0)
    hi = np.quantile(latents, 1 - (1 - coverage) / 2, axis=0)
    return lo.astype(np.float32), hi.astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Score network (denoising score matching) -- the GEN-CIM Phase 3b component
# ---------------------------------------------------------------------------

class ScoreNet(nn.Module):
    """s_theta(z, sigma) ~= grad_z log p_sigma(z). Sigma-conditioned MLP."""

    def __init__(self, dz: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dz + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dz),
        )

    def forward(self, z, sigma):
        s = torch.log(sigma).reshape(-1, 1)
        return self.net(torch.cat([z, s], dim=-1)) / sigma.reshape(-1, 1)


def train_scorenet(
    latents: np.ndarray,
    quality: np.ndarray,          # higher = better; e.g. -cost
    beta_s: float = 2.0,          # sharpness of the quality reweighting
    sigma_min: float = 0.05,
    sigma_max: float = 1.0,
    epochs: int = 300,
    batch: int = 256,
    lr: float = 1e-3,
    hidden: int = 512,
    seed: int = 0,
) -> ScoreNet:
    """DSM objective, GEN-CIM eq. (9):
           L = E[ sigma^2 * || s(z + sigma*eps, sigma) + eps/sigma ||^2 ]
       with samples drawn using importance weights ~ exp(beta_s * normalised
       quality), so capacity is spent on the good part of the manifold.
    """
    torch.manual_seed(seed)
    dev = DEVICE
    Z = torch.as_tensor(latents, dtype=torch.float32, device=dev)
    q = np.asarray(quality, dtype=np.float64)
    q = (q - q.mean()) / (q.std() + 1e-8)
    w = np.exp(beta_s * q)
    # Weights and multinomial stay on CPU: MPS has had correctness issues with
    # torch.multinomial, and this is a negligible cost either way.
    w = torch.as_tensor(w / w.sum(), dtype=torch.float32, device="cpu")
    gen = _cpu_gen(seed)

    net = ScoreNet(Z.shape[1], hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        idx = torch.multinomial(w, batch, replacement=True, generator=gen).to(dev)
        z = Z[idx]
        u = torch.rand(batch, generator=gen, device="cpu").to(dev)
        sigma = sigma_min * (sigma_max / sigma_min) ** u          # log-uniform
        eps = torch.randn_like(z)
        zt = z + sigma[:, None] * eps
        pred = net(zt, sigma)
        loss = ((sigma[:, None] ** 2) * (pred + eps / sigma[:, None]) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


@torch.no_grad()
def annealed_langevin(
    net: ScoreNet, n: int, dz: int, steps: int = 60, eta: float = 0.01,
    sigma_hi: float = 1.0, sigma_lo: float = 0.05, rng_seed: int = 0,
    box: Optional[tuple] = None,
) -> np.ndarray:
    """GEN-CIM eq. (10):  z <- z + (eta/2) s(z, sigma_t) + sqrt(eta) * eps

    Note this is the standard Langevin step with step size eps = eta/2, since
    x + eps*grad + sqrt(2*eps)*noise  ==  x + (eta/2)*grad + sqrt(eta)*noise.

    Costs ZERO decoder calls. That is the point: the warm start is free in the
    budget that actually matters.
    """
    dev = next(net.parameters()).device
    g = _cpu_gen(rng_seed)
    z = _randn((n, dz), g, dev)
    lo_t = hi_t = None
    if box is not None:
        lo_t = torch.as_tensor(box[0], dtype=torch.float32, device=dev)
        hi_t = torch.as_tensor(box[1], dtype=torch.float32, device=dev)
    for t in range(steps):
        sigma = sigma_hi + (sigma_lo - sigma_hi) * (t / max(1, steps - 1))
        s = net(z, torch.full((n,), sigma, device=dev))
        z = z + 0.5 * eta * s + math.sqrt(eta) * _randn(z.shape, g, dev)
        if lo_t is not None:
            z = torch.clamp(z, lo_t, hi_t)
    return z.detach().to("cpu").numpy()


# ---------------------------------------------------------------------------
# 6. Normalizing-flow prior -- the cheap competitor to the score model
# ---------------------------------------------------------------------------

class Coupling(nn.Module):
    def __init__(self, dz: int, hidden: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(dz, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * dz),
        )
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def _st(self, x):
        s, t = self.net(x * self.mask).chunk(2, dim=-1)
        s = torch.tanh(s) * 2.0
        return s * (1 - self.mask), t * (1 - self.mask)

    def forward(self, z):                     # data -> base
        s, t = self._st(z)
        u = z * self.mask + (1 - self.mask) * ((z - t) * torch.exp(-s))
        return u, -s.sum(-1)

    def inverse(self, u):                     # base -> data
        s, t = self._st(u)
        return u * self.mask + (1 - self.mask) * (u * torch.exp(s) + t)


class FlowPrior(nn.Module):
    def __init__(self, dz: int, layers: int = 6, hidden: int = 256):
        super().__init__()
        masks = []
        for i in range(layers):
            m = torch.zeros(dz)
            m[i % 2::2] = 1.0
            masks.append(m)
        self.layers = nn.ModuleList([Coupling(dz, hidden, m) for m in masks])
        self.dz = dz

    def log_prob(self, z):
        ld = torch.zeros(len(z), device=z.device)
        for layer in self.layers:
            z, d = layer(z)
            ld = ld + d
        base = -0.5 * (z ** 2).sum(-1) - 0.5 * self.dz * math.log(2 * math.pi)
        return base + ld

    @torch.no_grad()
    def sample(self, n: int, gen=None):
        dev = next(self.parameters()).device
        u = _randn((n, self.dz), gen or _cpu_gen(0), dev)
        for layer in reversed(self.layers):
            u = layer.inverse(u)
        return u


def train_flow(latents: np.ndarray, epochs: int = 800, batch: int = 256,
               lr: float = 1e-3, layers: int = 6, seed: int = 0) -> FlowPrior:
    torch.manual_seed(seed)
    dev = DEVICE
    Z = torch.as_tensor(latents, dtype=torch.float32, device=dev)
    flow = FlowPrior(Z.shape[1], layers=layers).to(dev)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    gen = _cpu_gen(seed)
    for _ in range(epochs):
        idx = torch.randint(0, len(Z), (min(batch, len(Z)),),
                            generator=gen, device="cpu").to(dev)
        loss = -flow.log_prob(Z[idx]).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
        opt.step()
    return flow.eval()


# ---------------------------------------------------------------------------
# 7. Samplers. Every one takes (BudgetedDecoder, rng, **cfg) and runs to budget.
# ---------------------------------------------------------------------------

def _clip(Z, box):
    return Z if box is None else np.clip(Z, box[0], box[1])


def sampler_random(d: BudgetedDecoder, rng, pop=600, box=None, **_):
    while not d.exhausted:
        d(_clip(rng.standard_normal((pop, d.dz)).astype(np.float32), box))


def _de_loop(d: BudgetedDecoder, rng, P, costs, CR=0.95, F=0.3, box=None):
    """DE/rand/1/bin, batched. CVAE-Opt used pop=600, CR=0.95, F=0.3."""
    pop = len(P)
    while not d.exhausted:
        a, b, c = (rng.integers(0, pop, pop) for _ in range(3))
        V = P[a] + F * (P[b] - P[c])
        cross = rng.random((pop, d.dz)) < CR
        jrand = rng.integers(0, d.dz, pop)
        cross[np.arange(pop), jrand] = True
        U = _clip(np.where(cross, V, P).astype(np.float32), box)
        cu = d(U)
        better = cu < costs
        P[better], costs[better] = U[better], cu[better]
    return P, costs


def sampler_de(d: BudgetedDecoder, rng, pop=600, box=None, **cfg):
    if box is None:
        P = rng.standard_normal((pop, d.dz)).astype(np.float32)
    else:
        P = rng.uniform(box[0], box[1], (pop, d.dz)).astype(np.float32)
    _de_loop(d, rng, P, d(P), box=box, **cfg)


def sampler_langevin_de(d: BudgetedDecoder, rng, scorenet: ScoreNet, pop=600,
                        lan_steps=60, eta=0.01, box=None, **cfg):
    """GEN-CIM design: score-model warm start (free), then evolutionary search."""
    P = annealed_langevin(scorenet, pop, d.dz, steps=lan_steps, eta=eta,
                          rng_seed=int(rng.integers(1 << 30)), box=box)
    P = _clip(P.astype(np.float32), box)
    _de_loop(d, rng, P, d(P), box=box, **cfg)


def sampler_flow_de(d: BudgetedDecoder, rng, flow: FlowPrior, pop=600, box=None, **cfg):
    """Learned prior without any MCMC: sample the flow, then the same DE."""
    g = _cpu_gen(int(rng.integers(1 << 30)))
    P = _clip(flow.sample(pop, g).to("cpu").numpy().astype(np.float32), box)
    _de_loop(d, rng, P, d(P), box=box, **cfg)


def sampler_metropolis(d: BudgetedDecoder, rng, chains=600, lam=5.0,
                       sigma_prop=0.1, gamma=0.379, log_prior=None, box=None,
                       adapt=True, target_accept=0.3, adapt_every=10, **_):
    """Cost-tilted Metropolis (LGS design).

    Target:   pi(z) ~ prior(z) * exp(-lam * cost(decode(z)))
    Proposal: z' = z + gamma * (z^{i1} - z^{i2}) + sigma_prop * eps   (symmetric)
    Accept:   min(1, exp(-lam * (C' - C)) * prior(z') / prior(z))

    BUDGET ALLOCATION WARNING. At a fixed decode budget, chains and iterations
    trade off directly: `iterations = budget / chains`. DE improves by
    selection every generation so it tolerates a large population and few
    generations. A Markov chain does not: 60 steps is not a chain, it is noise.
    Measured on the self-test (budget 12000, n=50, DE reference 16.21):

        chains=200, 60 iters   -> 21.41
        chains=50,  240 iters  -> 20.31
        chains=20,  600 iters  -> 19.83
        chains=8,   1500 iters -> 18.56

    So do NOT set chains = DE population and call it a fair comparison. Tune
    chains per arm, or you will conclude MCMC does not work when what you
    measured was a chain that never mixed. Use `tune_metropolis`.

    `lam` is the other high-leverage knob. Sweep both.
    """
    if log_prior is None:
        log_prior = lambda Z: -0.5 * (Z.astype(np.float64) ** 2).sum(-1)

    Z = _clip(rng.standard_normal((chains, d.dz)).astype(np.float32), box)
    C = d(Z)
    lp = log_prior(Z)
    n_acc = n_prop = 0
    log_s, log_g = math.log(sigma_prop), math.log(max(gamma, 1e-6))
    win_acc, win_n, it = 0, 0, 0

    while not d.exhausted:
        i1, i2 = rng.integers(0, chains, chains), rng.integers(0, chains, chains)
        Zp = _clip((Z + math.exp(log_g) * (Z[i1] - Z[i2])
                    + math.exp(log_s) * rng.standard_normal((chains, d.dz))
                    ).astype(np.float32), box)
        Cp = d(Zp)
        lpp = log_prior(Zp)
        log_alpha = -lam * (Cp - C) + (lpp - lp)
        acc = np.log(rng.random(chains) + 1e-300) < log_alpha
        acc &= np.isfinite(Cp)
        Z[acc], C[acc], lp[acc] = Zp[acc], Cp[acc], lpp[acc]
        k = int(np.isfinite(Cp).sum())
        n_acc += int(acc.sum()); n_prop += k
        win_acc += int(acc.sum()); win_n += k
        it += 1

        # Robbins-Monro adaptation of the proposal scale. Shrinking step size
        # keeps this a valid adaptive MCMC scheme rather than an ad hoc hack.
        if adapt and it % adapt_every == 0 and win_n > 0:
            rate = 0.5 / math.sqrt(1 + it / adapt_every)
            delta = rate * ((win_acc / win_n) - target_accept)
            log_s = float(np.clip(log_s + delta, math.log(1e-4), math.log(5.0)))
            log_g = float(np.clip(log_g + delta, math.log(1e-4), math.log(1.0)))
            win_acc = win_n = 0
    # Acceptance rate is the first thing to look at when this arm underperforms.
    # Near 0   -> lam too large or proposal too wide: the chain never moves.
    # Near 1   -> lam too small: this is random search with extra steps.
    # 0.2-0.5 and still bad -> the chain IS moving and the latent space has no
    #                          exploitable local structure. That is a finding,
    #                          not a bug. Run landscape_check() to confirm.
    d.diagnostics = {"accept_rate": n_acc / max(1, n_prop),
                     "final_sigma_prop": math.exp(log_s),
                     "final_gamma": math.exp(log_g)}


SAMPLERS: Dict[str, Callable] = {
    "random": sampler_random,
    "de": sampler_de,
    "langevin_de": sampler_langevin_de,
    "flow_de": sampler_flow_de,
    "metropolis": sampler_metropolis,
}


# ---------------------------------------------------------------------------
# 8. Stage 0: does the latent space have exploitable local structure?
# ---------------------------------------------------------------------------

def landscape_check(decoder: Decoder, rng, n_ref=1000, per_shell=100,
                    radii=(0.1, 0.2, 0.5, 2.0, 5.0), box=None):
    """Replicates CVAE-Opt's search-space structure experiment.

    Sample n_ref latents, take the best as a reference, then sample shells at
    increasing Euclidean distance from it and report the mean cost penalty.

    A monotone increasing curve means neighbouring latents decode to similar
    quality, i.e. the space is smooth and a local sampler has something to
    exploit. A flat curve means it does not, and no proposal distribution will
    rescue you. RUN THIS BEFORE THE MAIN EXPERIMENT.
    """
    Z = _clip(rng.standard_normal((n_ref, decoder.dz)).astype(np.float32), box)
    c = decoder.decode(Z)
    z_ref, c_ref = Z[int(np.argmin(c))], float(c.min())

    out = []
    for r in radii:
        eps = rng.standard_normal((per_shell, decoder.dz))
        eps /= np.linalg.norm(eps, axis=1, keepdims=True)
        Zs = _clip((z_ref + r * eps).astype(np.float32), box)
        out.append((r, float(np.mean(decoder.decode(Zs)) - c_ref)))
    return c_ref, out


def print_landscape(decoder, rng, **kw):
    c_ref, shells = landscape_check(decoder, rng, **kw)
    print(f"landscape check | reference cost {c_ref:.3f}")
    for r, d in shells:
        bar = "#" * int(max(0.0, d) * 4)
        print(f"  radius {r:>5.2f}   mean cost penalty {d:+7.3f}  {bar}")
    span = shells[-1][1] - shells[0][1]
    verdict = "SMOOTH (local search viable)" if span > 0.25 * abs(shells[-1][1]) + 1e-9 \
        else "FLAT (local samplers will not help)"
    print(f"  verdict: {verdict}")
    return shells


def tune_metropolis(decoder, budget, lams=(1, 5, 20, 100), chain_counts=(8, 20, 50, 200),
                    seeds=2, verbose=True, **cfg):
    """Joint sweep over the two knobs that decide whether this arm works at all.

    Run this on VALIDATION instances, pick the best cell, then evaluate that
    single configuration on the test set. Tuning on test and reporting the max
    is how you get a result that does not replicate.
    """
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


# ---------------------------------------------------------------------------
# 9. Runner
# ---------------------------------------------------------------------------

#: arms that consume a prior fitted on decoded solutions, so must declare
#: how many decodes went into fitting it
PRIOR_ARMS = {"langevin_de", "flow_de"}


def run_arm(name: str, decoder: Decoder, budget: int, seed: int,
            setup_evals: int = 0, amortize_over: int = 1, **cfg):
    """Run one arm to exhaustion of `budget` TOTAL decodes, setup included.

    `setup_evals` is the number of decoder calls spent building this arm's
    prior (the solution pool the score net or flow was trained on). It is
    charged up front. Arms without a learned prior pass 0.
    """
    if name in PRIOR_ARMS and setup_evals == 0 and amortize_over == 1:
        raise ValueError(
            f"arm '{name}' uses a prior fitted on decoded solutions but declared "
            f"setup_evals=0. Pass the pool size, or pass amortize_over=<n> if the "
            f"prior is trained offline and reused across n instances. Leaving this "
            f"unpriced silently hands the arm free decodes and invalidates the "
            f"comparison.")

    d = BudgetedDecoder(decoder, budget)
    d.charge_setup(setup_evals, amortize_over)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    SAMPLERS[name](d, rng, **cfg)
    return {"arm": name, "seed": seed, "best": d.best, "evals": d.used,
            "setup": d.setup_charged, "seconds": time.time() - t0,
            "trace": d.trace, "diagnostics": d.diagnostics}


def build_pool(d: BudgetedDecoder, rng, size: int, keep_frac: float = 0.5,
               box=None):
    """Build a training pool THROUGH the budgeted decoder, so the decodes are
    counted automatically. Use this instead of calling decoder.decode directly
    when the pool feeds a prior that will be evaluated in the same comparison.

    Returns (latents, quality) of the best `keep_frac` fraction, mirroring
    GEN-CIM Phase 2 which keeps only high-quality candidates.
    """
    Z = _clip(rng.standard_normal((size, d.dz)).astype(np.float32), box)
    c = d(Z)
    keep = np.argsort(c)[: max(1, int(size * keep_frac))]
    return Z[keep], -c[keep]


def evals_to_reach(trace, target) -> Optional[int]:
    for e, b in trace:
        if b <= target:
            return e
    return None


def summarise(results: List[dict], arms: List[str]) -> str:
    by_inst: Dict[int, Dict[str, List[float]]] = {}
    setup: Dict[str, int] = {}
    for r in results:
        by_inst.setdefault(r["instance"], {}).setdefault(r["arm"], []).append(r["best"])
        setup[r["arm"]] = r.get("setup", 0)

    # Per-instance reference = the best *mean-over-seeds* any arm achieved, so
    # a lucky single seed cannot set the bar.
    means = {i: {arm: float(np.mean(v)) for arm, v in a.items()} for i, a in by_inst.items()}
    ref = {i: min(m.values()) for i, m in means.items()}

    lines = [f"{'arm':<14}{'mean cost':>11}{'mean gap':>10}{'std gap':>9}{'wins':>7}{'setup':>9}"]
    lines.append("-" * 60)
    rows = []
    for arm in arms:
        gaps, costs, wins = [], [], 0
        for i, m in means.items():
            if arm not in m:
                continue
            costs.append(m[arm])
            gaps.append(100.0 * (m[arm] / ref[i] - 1.0))
            if abs(m[arm] - ref[i]) < 1e-9:
                wins += 1
        if costs:
            rows.append((arm, np.mean(costs), np.mean(gaps), np.std(gaps), wins))
    for arm, c, g, s, w in sorted(rows, key=lambda r: r[2]):
        lines.append(f"{arm:<14}{c:>11.4f}{g:>9.3f}%{s:>8.3f}%{w:>7d}{setup.get(arm,0):>9d}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 9. Self-test on the random-keys decoder
# ---------------------------------------------------------------------------

def selftest(n=50, instances=5, seeds=3, budget=30000, pop=300, lam=5.0,
             pool_size=4000, out=None):
    print(f"self-test | n={n} instances={instances} seeds={seeds} "
          f"budget={budget} device={DEVICE}")
    print("NOTE: random-keys decoder. Validates the samplers, does NOT predict\n"
          "      behaviour on a learned CVAE latent space.\n")

    arms = ["random", "de", "flow_de", "langevin_de", "metropolis"]
    results = []

    # Stage 0 first, on instance 0 only. If this says FLAT, expect the local
    # samplers to fail and do not read that as an implementation bug.
    print_landscape(RandomKeysDecoder(make_instance(n, np.random.default_rng(1000))),
                    np.random.default_rng(0))
    print()

    for inst_id in range(instances):
        rng = np.random.default_rng(1000 + inst_id)
        inst = make_instance(n, rng)
        dec = RandomKeysDecoder(inst)

        # Build a training pool the way GEN-CIM Phase 2 does: sample, keep the
        # good half, then fit the generative priors on those latents only.
        pool = rng.standard_normal((pool_size, dec.dz)).astype(np.float32)
        pcost = dec.decode(pool)
        keep = np.argsort(pcost)[: pool_size // 2]
        lat, qual = pool[keep], -pcost[keep]

        scorenet = train_scorenet(lat, qual, epochs=300, seed=inst_id)
        flow = train_flow(lat, epochs=600, seed=inst_id)

        cfg = {
            "random": dict(pop=pop),
            "de": dict(pop=pop),
            "flow_de": dict(pop=pop, flow=flow),
            "langevin_de": dict(pop=pop, scorenet=scorenet),
            # Chains deliberately NOT set to `pop`: see the budget-allocation
            # warning in sampler_metropolis. Aim for >=1000 chain iterations.
            "metropolis": dict(chains=max(8, budget // 1500), lam=lam, sigma_prop=0.1),
        }
        for seed in range(seeds):
            for arm in arms:
                # The pool is charged to the arms whose prior it trained. Set
                # amortize_over to >1 only if the prior is genuinely reused.
                su = pool_size if arm in PRIOR_ARMS else 0
                r = run_arm(arm, dec, budget, seed=seed * 97 + inst_id,
                            setup_evals=su, **cfg[arm])
                r["instance"] = inst_id
                results.append(r)
        print(f"  instance {inst_id}: " + "  ".join(
            f"{a}={min(r['best'] for r in results if r['instance']==inst_id and r['arm']==a):.3f}"
            for a in arms))

    print("\n" + summarise(results, arms))
    ar = [r["diagnostics"]["accept_rate"] for r in results
          if r["arm"] == "metropolis" and "accept_rate" in r["diagnostics"]]
    if ar:
        print(f"\nmetropolis mean acceptance rate: {np.mean(ar):.3f}"
              f"   (target roughly 0.2-0.5; see notes in sampler_metropolis)")
    if out:
        with open(out, "w") as f:
            json.dump([{k: v for k, v in r.items() if k != "trace"} for r in results], f, indent=1)
        print(f"\nwrote {out}")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="selftest",
                   choices=["selftest", "landscape", "tune"])
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--instances", type=int, default=5)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--budget", type=int, default=30000)
    p.add_argument("--pop", type=int, default=300)
    p.add_argument("--lam", type=float, default=5.0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    if a.mode == "selftest":
        selftest(a.n, a.instances, a.seeds, a.budget, a.pop, a.lam, out=a.out)
    elif a.mode == "landscape":
        dec = RandomKeysDecoder(make_instance(a.n, np.random.default_rng(1000)))
        print_landscape(dec, np.random.default_rng(0))
    elif a.mode == "tune":
        dec = RandomKeysDecoder(make_instance(a.n, np.random.default_rng(1000)))
        print(f"tuning metropolis | budget={a.budget}")
        tune_metropolis(dec, a.budget, seeds=a.seeds)


if __name__ == "__main__":
    main()
