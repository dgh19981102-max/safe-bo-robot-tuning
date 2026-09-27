"""E2: payload changes during operation (20 seeds).

The robot handles parts of 1.0 -> 2.5 -> 1.75 -> 0.5 -> 2.0 kg, with 10 tuning trials
allowed per change.
  Fixed gains  : keep the gains tuned at 1.0 kg (no re-tuning)
  Cold safe BO : re-tune from the conservative seed at every change
  Warm safe BO : contextual safe BO re-using all earlier trials (payload as GP input)
"""
import time
import numpy as np
from common import save, SEED_U, N_SEEDS, trace
from autotune import optimizers as O
from autotune.task import evaluate
from autotune.controller import to_gains

SEQ = [1.0, 2.5, 1.75, 0.5, 2.0]
B = 10
out = {"sequence": SEQ, "budget": B, "cold": [], "warm": [], "fixed": []}
t = time.time()
for s in range(N_SEEDS):
    for mode in ["cold", "warm"]:
        hist, stages = [], []
        for k, pl in enumerate(SEQ):
            rng = np.random.default_rng(1000 * s + k)
            e = O.Experiment(pl, 20_000 + 1000 * s + k)
            if mode == "cold" or k == 0:
                O.safe_bayes_opt(e, B, rng, SEED_U)
            else:
                O.contextual_safe_bayes_opt(e, B, rng, SEED_U, hist)
            hist = hist + e.log
            stages.append(trace(e.log))
            if mode == "cold" and k == 0:
                safe0 = [r for r in e.log if r["safe"]]
                g1 = to_gains(min(safe0, key=lambda r: r["cost"])["u"])
        out[mode].append(stages)
    # fixed gains from stage 1 (cold run), evaluated at every payload
    out["fixed"].append([{k: evaluate(g1, pl, seed=s)[k] for k in ("cost", "safety", "safe")}
                         for pl in SEQ])
    print("seed", s, f"{time.time() - t:.0f}s", flush=True)
save("e2_payload_sequence.json", out)
