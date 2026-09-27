"""Publication figures and summary table from the experiment results.

Run after e0/e1/e2:  python experiments/make_figures.py
Writes docs/figures/*.png and results/summary.json
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from common import ROOT, load, save, SEED_GAINS

from autotune.task import evaluate, SAFE_MAX_ERR_DEG
from autotune.controller import to_gains

OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

TEAL, BLUE, GREY, AMBER, RED = "#1f6f7a", "#5b8fc7", "#9a9a9a", "#e0a13a", "#c0504d"
COL = {"Random search": GREY, "CMA-ES": AMBER, "BO (EI)": BLUE, "Safe BO": TEAL}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.facecolor": "#fbfaf6",
    "axes.edgecolor": "#c9c9c9", "axes.grid": True, "grid.color": "#e3e1dc",
    "grid.linestyle": "--", "axes.titlesize": 9.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.axisbelow": True,
})

ref = load("reference.json")
e1 = load("e1_sample_efficiency.json")
e2 = load("e2_payload_sequence.json")
summary = {}


def best_safe_curve(run):
    c = np.where(np.array(run["safe"]), np.array(run["cost"]), np.inf)
    return np.minimum.accumulate(c)


# ------------------------------------------------------------------ E1 statistics
ref1 = ref["1.0"]["cost"]
seed_cost = np.mean([evaluate(SEED_GAINS, 1.0, seed=s)["cost"] for s in range(10)])
summary["seed_cost_1kg"] = seed_cost
summary["reference_cost_1kg"] = ref1
summary["e1"] = {}
for m, runs in e1.items():
    curves = np.array([best_safe_curve(r) for r in runs])
    unsafe = np.array([np.sum(~np.array(r["safe"])) for r in runs])
    worst = np.array([np.max(r["safety"]) for r in runs])
    gap15 = (curves[:, 14] - ref1) / ref1 * 100
    gap50 = (curves[:, -1] - ref1) / ref1 * 100
    summary["e1"][m] = dict(unsafe_mean=float(unsafe.mean()), unsafe_max=int(unsafe.max()),
                            runs_without_unsafe=int(np.sum(unsafe == 0)),
                            worst_safety_median=float(np.median(worst)),
                            gap15_median=float(np.median(gap15)), gap50_median=float(np.median(gap50)),
                            best15_median=float(np.median(curves[:, 14])),
                            best50_median=float(np.median(curves[:, -1])))

# ------------------------------------------------------------------ Figure 1: overview
fig, ax = plt.subplots(1, 3, figsize=(12, 3.7))
x = np.arange(1, 51)
for m, runs in e1.items():
    curves = np.array([best_safe_curve(r) for r in runs])
    curves = np.where(np.isfinite(curves), curves, np.nan)
    med = np.nanmedian(curves, 0)
    lo, hi = np.nanpercentile(curves, 25, 0), np.nanpercentile(curves, 75, 0)
    ax[0].plot(x, med, color=COL[m], lw=2.2 if m == "Safe BO" else 1.6, label=m)
    ax[0].fill_between(x, lo, hi, color=COL[m], alpha=0.15, lw=0)
ax[0].axhline(ref1, color="#555", lw=0.9, ls=":")
ax[0].text(50, ref1 - 0.03, "reference optimum", ha="right", va="top", fontsize=7.5, color="#555")
ax[0].axhline(seed_cost, color="#555", lw=0.9, ls="--")
ax[0].text(49, seed_cost + 0.02, "commissioning gains", ha="right", va="bottom", fontsize=7.5, color="#555")
ax[0].set(xlim=(1, 50), ylim=(ref1 * 0.9, seed_cost * 1.15), xlabel="Experiments on the robot",
          title="Best safe cost found (median, IQR, 20 runs)")
ax[0].legend(fontsize=7.5, frameon=False, loc="lower left", bbox_to_anchor=(0.02, 0.08))

names = list(e1.keys())
rng = np.random.default_rng(0)
for i, m in enumerate(names):
    s = np.concatenate([r["safety"] for r in e1[m]])
    jit = i + rng.uniform(-0.28, 0.28, len(s))
    ax[1].scatter(jit, s, s=5, color=COL[m], alpha=0.35, lw=0)
    frac = np.mean(s > 1) * 100
    ax[1].text(i, 13, f"{frac:.0f}%\nunsafe", ha="center", fontsize=7.5, color="#444")
ax[1].axhline(1.0, color=RED, lw=1.2)
ax[1].text(-0.45, 1.08, "safety limit", ha="left", fontsize=7.5, color=RED)
ax[1].set_yscale("log")
ax[1].set(ylim=(0.12, 25), xticks=range(len(names)), title="Safety index of every trial (1000 per method)")
ax[1].set_xticklabels([n.replace(" search", "\nsearch") for n in names], fontsize=8)

seq = e2["sequence"]
labels = [f"{p:g} kg" for p in seq]
gap = {}
for mode in ["cold", "warm"]:
    g = []
    for k, pl in enumerate(seq):
        r0 = ref[str(pl)]["cost"]
        vals = [(best_safe_curve(run[k])[9] - r0) / r0 * 100 for run in e2[mode]]
        g.append(np.median(vals))
    gap[mode] = np.array(g)
fixed = np.array([[f["cost"] for f in run] for run in e2["fixed"]])
fixed_safe = np.array([[f["safe"] for f in run] for run in e2["fixed"]])
gap["fixed"] = np.array([np.median((fixed[:, k] - ref[str(pl)]["cost"]) / ref[str(pl)]["cost"] * 100)
                         for k, pl in enumerate(seq)])
w = 0.27
xs = np.arange(len(seq))
ax[2].bar(xs - w, gap["fixed"], w, color=GREY, label="keep 1 kg gains")
ax[2].bar(xs, gap["cold"], w, color=BLUE, label="safe BO, cold start")
ax[2].bar(xs + w, gap["warm"], w, color=TEAL, label="safe BO, warm start")
for k in range(len(seq)):
    pu = np.mean(~fixed_safe[:, k]) * 100
    if pu > 0:
        ax[2].text(xs[k] - w, gap["fixed"][k] + 1, f"{pu:.0f}% runs\nunsafe", ha="center", va="bottom",
                   fontsize=7, color=RED)
ax[2].set_xticks(xs, labels)
ax[2].set(title="Payload changes: gap to optimum after 10 trials (%)", xlabel="Payload sequence →")
ax[2].set_ylim(0, max(gap["fixed"].max(), gap["cold"].max()) * 1.45)
ax[2].legend(fontsize=7.5, frameon=False, loc="upper left", ncol=1)
fig.suptitle("Safe Bayesian Optimisation for Robot-Arm Controller Tuning", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_overview.png"), dpi=180)
plt.close(fig)
summary["e2"] = {m: list(map(float, v)) for m, v in gap.items()}
summary["e2_fixed_unsafe_pct"] = list(map(float, np.mean(~fixed_safe, 0) * 100))
summary["e2_unsafe_trials"] = {m: int(sum(np.sum(~np.array(st["safe"])) for run in e2[m] for st in run))
                               for m in ["cold", "warm"]}
summary["e2_trials_total"] = {m: int(sum(len(st["safe"]) for run in e2[m] for st in run))
                              for m in ["cold", "warm"]}

# ------------------------------------------------------------------ Figure 2: responses
runs = e1["Safe BO"]
best_run = sorted(runs, key=lambda r: best_safe_curve(r)[-1])[len(runs) // 2]   # median run
i_best = int(np.argmin(np.where(best_run["safe"], best_run["cost"], np.inf)))
g_tuned = to_gains(best_run["u"][i_best])
a = evaluate(SEED_GAINS, 1.0, seed=7, return_traj=True)
b = evaluate(g_tuned, 1.0, seed=7, return_traj=True)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
for k, (j, name) in enumerate([(1, "joint 1"), (3, "joint 2")]):
    ta, tb = a["traj"], b["traj"]
    ax[k].plot(ta[:, 0], np.degrees(ta[:, j] - ta[:, j + 1]), color=GREY, lw=1.4, ls="--", label="commissioning gains")
    ax[k].plot(tb[:, 0], np.degrees(tb[:, j] - tb[:, j + 1]), color=TEAL, lw=1.6, label="safe BO tuned")
    ax[k].axvspan(1.4, 1.5, color=AMBER, alpha=0.18, lw=0)
    ax[k].set(title=f"Tracking error, {name} (deg)", xlabel="Time (s)", xlim=(0, 2))
ax[0].text(1.45, ax[0].get_ylim()[1] * 0.85, "push", ha="center", fontsize=7.5, color="#8a6d2b")
ax[0].legend(fontsize=7.5, frameon=False, loc="lower left")
ax[2].plot(a["traj"][:, 0], a["traj"][:, 5], color=GREY, lw=1, ls="--")
ax[2].plot(b["traj"][:, 0], b["traj"][:, 5], color=TEAL, lw=1.2)
ax[2].set(title="Joint 1 torque (N m)", xlabel="Time (s)", xlim=(0, 2))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_response.png"), dpi=180)
plt.close(fig)
summary["response"] = dict(seed=dict(rmse=a["rmse_deg"], max_err=a["max_err_deg"], chatter=a["chatter"]),
                           tuned=dict(rmse=b["rmse_deg"], max_err=b["max_err_deg"], chatter=b["chatter"],
                                      gains=list(map(float, g_tuned))))

# ------------------------------------------------------------------ Figure 3: payload sequence trace
fig, ax = plt.subplots(1, 1, figsize=(12, 3.2))
B = e2["budget"]
for mode, col in [("cold", BLUE), ("warm", TEAL)]:
    tr = np.array([np.concatenate([np.minimum.accumulate(np.where(st["safe"], st["cost"], np.inf))
                                   for st in run]) for run in e2[mode]])
    med = np.median(tr, 0)
    ax.plot(np.arange(1, len(med) + 1), med, color=col, lw=2, label=f"safe BO, {mode} start")
ytop = ax.get_ylim()[1]
ax.set_ylim(ax.get_ylim()[0], ytop + 0.15)
for k, pl in enumerate(seq):
    ax.axvline(k * B + 0.5, color="#bbb", lw=0.8)
    ax.text(k * B + B / 2 + 0.5, ytop + 0.1, f"payload {pl:g} kg", ha="center", va="top", fontsize=8, color="#444")
    ax.hlines(ref[str(pl)]["cost"], k * B + 0.5, (k + 1) * B + 0.5, color="#555", lw=0.9, ls=":")
ax.set(xlim=(0.5, len(seq) * B + 0.5), xlabel="Experiments on the robot (10 allowed per payload change)",
       title="Best safe cost within each payload stage (median of 20 runs; dotted = reference optimum)")
ax.legend(fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0.21, 0.03))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_payload_sequence.png"), dpi=180)
plt.close(fig)

save("summary.json", summary)
import json
print(json.dumps(summary, indent=1))
