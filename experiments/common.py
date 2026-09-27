import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

from autotune.controller import to_unit  # noqa: E402

# Conservative gains an engineer would commission on day one: safe at every payload
# in 0.5-2.5 kg, but far from optimal. All safe methods start from here.
SEED_GAINS = [600, 900, 15, 300, 500, 6]
SEED_U = to_unit(SEED_GAINS)
N_SEEDS = 20


def save(name, obj):
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f)


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def trace(log):
    """Per-evaluation arrays: cost, safety, safe flag."""
    return dict(cost=[r["cost"] for r in log], safety=[r["safety"] for r in log],
                safe=[bool(r["safe"]) for r in log], u=[list(map(float, r["u"])) for r in log])
