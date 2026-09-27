"""Reference optimum per payload: long CMA-ES runs (unconstrained budget), then the
best gains are re-evaluated on 10 noise realisations. Used only to measure the gap
to optimum; a real robot could never afford this many trials."""
import numpy as np
from common import save, SEED_U
from autotune import optimizers as O
from autotune.task import evaluate
from autotune.controller import to_gains

PAYLOADS = [0.5, 1.0, 1.75, 2.0, 2.5]
ref = {}
for pl in PAYLOADS:
    best = None
    for rs in range(3):
        rng = np.random.default_rng(rs)
        e = O.Experiment(pl, 500 + rs)
        O.cmaes(e, 400, rng, SEED_U if rs == 0 else rng.random(6), sigma0=0.2)
        cand = sorted([r for r in e.log if r["safe"]], key=lambda r: r["cost"])[:5]
        for r in cand:
            m = np.mean([evaluate(to_gains(r["u"]), pl, seed=s)["cost"] for s in range(10)])
            if best is None or m < best[0]:
                best = (m, r["u"])
    ref[str(pl)] = dict(cost=float(best[0]), gains=list(map(float, to_gains(best[1]))))
    print(pl, ref[str(pl)])
save("reference.json", ref)
