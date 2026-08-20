"""Project B: budget reliability curve, P(efficient | c) over 10 seeds (LOCAL).

For each budget coefficient c in {1.5, 2, 2.5, 3} (T = ceil(c*log2 n)), each of
10 seeds is classified by the slope of T*(n) vs n from loops_vs_length.json:
  efficient   slope <= 0.6
  sequential  slope >= 0.8   (the T* = n scan)
  mixed       in between
Panel (a): stacked class counts per c with the P(efficient) curve and Wilson
95% intervals.  Panel (b): reach (largest length solved at any tested T) per
seed, the orthogonal seed-lottery axis; slope class does not determine reach.

Reads:  runs/s5_logc{1p5,2,2p5,3}_* (seed 0 = original curr32/s0 run)
Usage:  python scripts/plot_reliability_budget.py
Writes: figures/reliability_budget_s5.{png,pdf}
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_EFF, C_MIX, C_SEQ = "#2b8a3e", "#e8a838", "#c0392b"
PT, MEAN = "#2b6ca3", "#16324f"


def _seeds(tag, first):
    return [f"runs/{first}"] + [f"runs/s5_logc{tag}_s{s}" for s in range(1, 10)]


RUNS = {
    1.5: _seeds("1p5", "s5_logc1p5_s0"),
    2.0: _seeds("2", "s5_logc2_curr32"),
    2.5: _seeds("2p5", "s5_logc2p5_s0"),
    3.0: _seeds("3", "s5_logc3_curr32"),
}


def curve_stats(path):
    o = json.load(open(os.path.join(path, "loops_vs_length.json")))
    cur = o["curve"][0] if isinstance(o["curve"], list) else o["curve"]
    solved = sorted((int(n), d["T_star"]) for n, d in cur.items()
                    if d["T_star"] is not None)
    pos = np.array([n for n, _ in solved], float)
    ts = np.array([t for _, t in solved], float)
    return float(np.polyfit(pos, ts, 1)[0]), int(pos.max())


def classify(slope):
    if slope <= 0.6:
        return "efficient"
    if slope >= 0.8:
        return "sequential"
    return "mixed"


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return ctr - hw, ctr + hw


def main():
    counts, reaches = {}, {}
    for c, paths in RUNS.items():
        st = [curve_stats(p) for p in paths]
        cls = [classify(s) for s, _ in st]
        counts[c] = {k: cls.count(k) for k in
                     ("efficient", "mixed", "sequential")}
        reaches[c] = [r for _, r in st]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    cs = list(RUNS)
    n_seeds = 10

    # (a) stacked class counts + P(efficient) with Wilson 95% intervals
    w = 0.28
    eff = np.array([counts[c]["efficient"] for c in cs], float)
    mix = np.array([counts[c]["mixed"] for c in cs], float)
    seq = np.array([counts[c]["sequential"] for c in cs], float)
    ax1.bar(cs, eff, w, color=C_EFF, alpha=0.75, label="efficient "
            "(slope$\\leq$0.6)")
    ax1.bar(cs, mix, w, bottom=eff, color=C_MIX, alpha=0.75, label="mixed")
    ax1.bar(cs, seq, w, bottom=eff + mix, color=C_SEQ, alpha=0.75,
            label="sequential (slope$\\geq$0.8)")
    lo, hi = zip(*[wilson(counts[c]["efficient"], n_seeds) for c in cs])
    p = eff / n_seeds
    ax1.errorbar(cs, p * n_seeds, yerr=[p * n_seeds - np.array(lo) * n_seeds,
                                        np.array(hi) * n_seeds - p * n_seeds],
                 fmt="o-", color=MEAN, lw=1.6, ms=5, capsize=3, zorder=5,
                 label="P(efficient), Wilson 95%")
    for c, k in zip(cs, eff.astype(int)):
        ax1.text(c, 10.25, f"{k}/10", ha="center", fontsize=8.5, color=MEAN)
    ax1.set_ylim(0, 11.6)
    ax1.set_yticks([0, 2, 4, 6, 8, 10])
    ax1.set_ylabel("seeds (of 10)")
    ax1.set_title("(a) algorithm class by budget", fontsize=10)
    ax1.legend(fontsize=7, loc="center left", framealpha=0.9)

    # (b) reach per seed (jittered) with per-c means
    for c in cs:
        vals = reaches[c]
        off = np.linspace(-0.12, 0.12, len(vals))
        ax2.plot(off + c, vals, "o", ms=5, color=PT, mec="white", mew=0.7,
                 alpha=0.85, zorder=3)
        ax2.hlines(np.mean(vals), c - 0.16, c + 0.16, color=MEAN, lw=2,
                   zorder=4)
    ax2.axhline(32, ls="--", color="#ff5555", lw=1.3)
    ax2.text(2.98, 32, "trained $n_{max}$", color="#ff5555", fontsize=8,
             ha="right", va="bottom")
    ax2.set_yticks([12, 16, 24, 32])
    ax2.set_ylim(8, 36)
    ax2.set_ylabel("reach (largest $n$ solved)")
    ax2.set_title("(b) reach per seed", fontsize=10)

    for ax in (ax1, ax2):
        ax.set_xlabel("budget coefficient $c$   ($T=\\lceil c\\log_2 n\\rceil$)")
        ax.set_xticks(cs)
        ax.grid(axis="y", alpha=0.3, lw=0.6)
        ax.set_axisbelow(True)

    fig.suptitle("Reliability of the induced algorithm on S5 (10 seeds per "
                 "budget): tight budgets always find the efficient loop; "
                 "slack budgets degrade to a seed lottery", fontsize=10.5,
                 y=1.04)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/reliability_budget_s5.{ext}", dpi=200,
                    bbox_inches="tight")
    print("wrote figures/reliability_budget_s5.png/.pdf")
    for c in cs:
        print(f"  c={c}: {counts[c]}  reach={sorted(reaches[c])}")


if __name__ == "__main__":
    main()
