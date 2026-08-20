"""Project B: training dynamics of the wavefront, sequential-then-compress (LOCAL).

Wavefront-probe verdicts (probe.py, S5 n=16) on intermediate checkpoints of the
c=2 log-budget run (runs/s5_logc2_curr32, seed 0): the model first lays down
the sequential scan (slope near 1 by step 8k, before the curriculum reaches
lengths the budget cannot scan), then compresses it into the efficient
wavefront (slope 0.53 by 40k) as the curriculum outgrows the budget.

  left axis  = slope of solve_loop vs position (1.0 = sequential diagonal)
  right axis = positions decodable at any loop (of 16)

Reads:  runs/s5_logc2_curr32/probe_dyn_<step>.json  (10 checkpoints)
Usage:  python scripts/plot_dynamics.py
Writes: figures/dynamics_s5_c2.{png,pdf}
"""
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "runs/s5_logc2_curr32"
SLOPE_C, SOLVED_C, REF = "#16324f", "#2b6ca3", "#ff5555"


def main():
    steps, slopes, solved = [], [], []
    for path in sorted(glob.glob(os.path.join(RUN, "probe_dyn_*.json"))):
        o = json.load(open(path))
        steps.append(int(re.search(r"probe_dyn_(\d+)\.json", path).group(1)))
        # early ckpts decode <3 positions: no slope is fittable (verdict {})
        slopes.append(o["verdict"].get("slope_vs_pos", np.nan))
        solved.append(sum(v is not None for v in o["solve_loop"].values()))
    order = np.argsort(steps)
    steps = np.array(steps)[order]
    slopes = np.array(slopes)[order]
    solved = np.array(solved)[order]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ok = ~np.isnan(slopes)
    ax.plot(steps[ok], slopes[ok], "o-", color=SLOPE_C, lw=1.8, ms=5,
            zorder=4, label="wavefront slope (probe, $n{=}16$)")
    ax.axhline(1.0, ls="--", color=REF, lw=1.2)
    ax.text(steps[0], 1.005, "sequential diagonal", color=REF, fontsize=8,
            va="bottom")
    ax.axhline(0.6, ls=":", color="#2b8a3e", lw=1.2)
    ax.text(steps[0], 0.605, "efficient threshold", color="#2b8a3e",
            fontsize=8, va="bottom")
    # curriculum reaches n_max = 32 at 60% of 40k steps
    ax.axvline(24000, color="#888888", lw=1.1, ls="-.")
    ax.text(24000 * 1.03, 0.15, "curriculum reaches $n_{max}$",
            color="#666666", fontsize=8, rotation=90, va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(steps)
    ax.set_xticklabels([f"{s//1000}k" if s >= 1000 else str(s)
                        for s in steps], fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("training step")
    ax.set_ylabel("slope of solve loop vs position")
    ax.set_ylim(0, 1.25)
    ax.grid(axis="y", alpha=0.3, lw=0.6)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    ax2.plot(steps, solved, "s--", color=SOLVED_C, lw=1.4, ms=4, alpha=0.8,
             label="positions decodable (of 16)")
    ax2.set_ylabel("positions decodable (of 16)", color=SOLVED_C)
    ax2.tick_params(axis="y", labelcolor=SOLVED_C)
    ax2.set_ylim(0, 17.5)
    ax2.set_yticks([0, 4, 8, 12, 16])

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left",
              bbox_to_anchor=(0.01, 0.30), framealpha=0.9)
    ax.set_title("Budget Pressure Breaks the Constant-Speed Computation Frontier comes first: under a $c{=}2$ log budget the "
                 "model learns the\nsequential scan, then compresses it into "
                 "the efficient wavefront", fontsize=10)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/dynamics_s5_c2.{ext}", dpi=200,
                    bbox_inches="tight")
    print("wrote figures/dynamics_s5_c2.png/.pdf")
    for s, sl, k in zip(steps, slopes, solved):
        print(f"  step {s:>6}: slope={sl:.2f} solved={k}/16")


if __name__ == "__main__":
    main()
