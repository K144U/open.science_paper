"""Project B: budget sweet-spot summary across the c-sweep (LOCAL).

Two panels over c in {1, 1.5, 2, 2.5, 3} (T budget = ceil(c*log2 n)), one dot
per seed, computed from each run's loops_vs_length.json:
  left  = linear slope of T*(n) vs n  (1.0 = sequential scan; ~0 = log-like)
  right = reach, the largest trained length solved (trained n_max = 32)

Reads:  runs/s5_logc1_{curr32,s1,s2} (3 seeds) and 10 seeds per c for
        c in {1.5, 2, 2.5, 3} (seed 0 = the original curr32/s0 run)
Usage:  python scripts/plot_csweep.py
Writes: figures/csweep_s5.{png,pdf}
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _seeds(tag, first):
    return [f"runs/{first}"] + [f"runs/s5_logc{tag}_s{s}" for s in range(1, 10)]


RUNS = {
    1.0: ["runs/s5_logc1_curr32", "runs/s5_logc1_s1", "runs/s5_logc1_s2"],
    1.5: _seeds("1p5", "s5_logc1p5_s0"),
    2.0: _seeds("2", "s5_logc2_curr32"),
    2.5: _seeds("2p5", "s5_logc2p5_s0"),
    3.0: _seeds("3", "s5_logc3_curr32"),
}
PT, MEAN, REF = "#2b6ca3", "#16324f", "#ff5555"


def curve_stats(path):
    o = json.load(open(os.path.join(path, "loops_vs_length.json")))
    cur = o["curve"][0] if isinstance(o["curve"], list) else o["curve"]
    solved = sorted((int(n), d["T_star"]) for n, d in cur.items()
                    if d["T_star"] is not None)
    pos = np.array([n for n, _ in solved], float)
    ts = np.array([t for _, t in solved], float)
    return float(np.polyfit(pos, ts, 1)[0]), int(pos.max())


def main():
    slopes, reaches = {}, {}
    for c, paths in RUNS.items():
        st = [curve_stats(p) for p in paths]
        slopes[c] = [s for s, _ in st]
        reaches[c] = [r for _, r in st]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax in (ax1, ax2):
        ax.set_xlabel("budget coefficient $c$   ($T=\\lceil c\\log_2 n\\rceil$)")
        ax.set_xticks(list(RUNS))
        ax.grid(axis="y", alpha=0.3, lw=0.6)
        ax.set_axisbelow(True)

    # jitter replicate dots so coincident seeds stay visible
    def dots(ax, data):
        for c, vals in data.items():
            off = (np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1
                   else [0.0])
            ax.plot(np.asarray(off) + c, vals, "o", ms=5, color=PT,
                    mec="white", mew=0.7, alpha=0.85, zorder=3)
            ax.hlines(np.mean(vals), c - 0.16, c + 0.16, color=MEAN, lw=2,
                      zorder=4)

    dots(ax1, slopes)
    ax1.axhline(1.0, ls="--", color=REF, lw=1.3)
    ax1.text(2.98, 1.0, "sequential scan ($T^*=n$)", color=REF, fontsize=8,
             ha="right", va="bottom")
    ax1.set_ylabel("slope of $T^*(n)$ vs $n$")
    ax1.set_ylim(0, 1.12)
    ax1.set_title("(a) algorithm efficiency", fontsize=10)

    dots(ax2, reaches)
    ax2.axhline(32, ls="--", color=REF, lw=1.3)
    ax2.text(2.98, 32, "trained $n_{max}$", color=REF, fontsize=8,
             ha="right", va="bottom")
    ax2.set_ylabel("reach (largest $n$ solved)")
    ax2.set_yticks([4, 8, 12, 16, 24, 32])
    ax2.set_ylim(0, 36)
    ax2.set_title("(b) reach", fontsize=10)

    fig.suptitle("The budget sweet spot on S5 (10 seeds per $c\\geq 1.5$, 3 at "
                 "$c{=}1$): tight-but-feasible budgets induce the efficient "
                 "loop; slack budgets become a seed lottery", fontsize=10.5,
                 y=1.04)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/csweep_s5.{ext}", dpi=200, bbox_inches="tight")
    print("wrote figures/csweep_s5.png/.pdf")
    for c in RUNS:
        print(f"  c={c}: slopes={np.round(slopes[c], 3).tolist()} "
              f"reach={reaches[c]}")


if __name__ == "__main__":
    main()
