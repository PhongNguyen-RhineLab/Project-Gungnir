import numpy as np, time
from calib2 import make_instance, _de, _eval_pop, MODES

POP=600
budgets=[60_000, 240_000, 960_000]
insts=[make_instance(100, 1000+k) for k in range(4)]

# random-search reference at the largest budget
rs={}
for name,mode in MODES.items():
    vals=[]
    for D,dem,Q in insts:
        rng=np.random.default_rng(3); best=1e18
        for _ in range(20):
            P=rng.random((10000,100))
            best=min(best, _eval_pop(P,D,dem,Q,mode).min())
        vals.append(best)
    rs[name]=np.mean(vals)

print(f"{'budget':>10s} {'bean':>9s} {'split':>9s}")
print("-"*31)
rows={}
for B in budgets:
    it=B//POP
    row=[]
    for name,mode in MODES.items():
        v=[_de(D,dem,Q,mode,100,POP,it,7,0.3,0.95) for D,dem,Q in insts]
        row.append(np.mean(v))
    rows[B]=row
    print(f"{B:>10,} {row[0]:9.3f} {row[1]:9.3f}", flush=True)
print("-"*31)
print(f"{'rand 200k':>10s} {rs['bean']:9.3f} {rs['split']:9.3f}")
print()
print("published CVRP100: LKH3 15.54 | CVAE-Opt-DE 15.75 | Opt-DE 23.97")
