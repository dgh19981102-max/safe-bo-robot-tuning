"""E1: sample efficiency and safety at the nominal 1 kg payload (20 seeds x 50 trials)."""
import time
import numpy as np
from common import save, SEED_U, N_SEEDS, trace
from autotune import optimizers as O

BUDGET = 50
methods = {
    "Random search": lambda e, rng: O.random_search(e, BUDGET, rng),
    "CMA-ES": lambda e, rng: O.cmaes(e, BUDGET, rng, SEED_U),
    "BO (EI)": lambda e, rng: O.bayes_opt(e, BUDGET, rng),
    "Safe BO": lambda e, rng: O.safe_bayes_opt(e, BUDGET, rng, SEED_U),
}
out = {}
for name, run in methods.items():
    t = time.time()
    out[name] = []
    for s in range(N_SEEDS):
        rng = np.random.default_rng(s)
        e = O.Experiment(1.0, 10_000 + s)
        run(e, rng)
        out[name].append(trace(e.log))
    print(name, f"{time.time() - t:.0f}s", flush=True)
save("e1_sample_efficiency.json", out)
