# gungnir

Tests whether a learned prior over the CVAE-Opt latent space improves latent search on CVRP, comparing arms at an equal decode budget.

Requirements: Python 3.10+, numpy, torch. scipy is not needed. numba is optional and only speeds up `calib` and `sweep`. You also need a checkout of CVAE-Opt:

```
git clone --depth 1 https://github.com/ahottung/CVAE-Opt.git
```

## Old command to new command

| before | now |
|---|---|
| `python3 latent_search.py --n 50 --budget 30000` | `python -m gungnir selftest --n 50 --budget 30000` |
| `python3 latent_search.py --mode landscape` | `python -m gungnir landscape` |
| `python3 latent_search.py --mode tune` | `python -m gungnir tune` |
| `python3 run_real.py --bench ...` | `python -m gungnir bench ...` |
| `python3 run_real.py --landscape ...` | `python -m gungnir landscape --real ...` |
| `python3 run_real.py ...` | `python -m gungnir run ...` (same flags) |
| `python3 analyze_results.py f.json` | `python -m gungnir analyze f.json --unit cell` (old numbers) |
| `python3 calib2.py` | `python -m gungnir calib` |
| `python3 sweep.py` | `python -m gungnir sweep` |

Every run flag from before is still there: `--repo --size --instances --start --seeds --budget --pop --pool --amortize --lam --chains --device --batch --arms --flow-epochs --score-epochs --keep-traces --out`. With the same flags, `run` and `selftest` give bit-identical results to the old scripts.

## Examples

Confirmation run, as before:

```
python -m gungnir run --repo ./CVAE-Opt --size 100 --start 20 --instances 24 --seeds 3 \
  --budget 180600 --pop 200 --pool 6000 --batch 256 --device mps \
  --arms de,flow_de,langevin_de --flow-epochs 400 --score-epochs 400 \
  --keep-traces --out confirm.json
```
Honest amortization test. The prior is fit once on held-out instances 900 to 923 and then reused frozen:

```
python -m gungnir run --repo ./CVAE-Opt --size 100 --start 20 --instances 24 --seeds 3 \
  --budget 12000 --pop 200 --pool 6000 --pool-mode shared --pool-start 900 --pool-instances 24 \
  --arms de,flow_de,langevin_de --keep-traces --out amortized.json
```

Analysis:

```
python -m gungnir analyze confirm.json --baseline flow_de
python -m gungnir analyze confirm.json --baseline de --curve
```
