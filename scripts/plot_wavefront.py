"""Project B centerpiece figure: prefix-state wavefront heatmaps (LOCAL).

Two panels from the Phase-2 probe (code/projB/probe.py) on S5, n=16:
  left  = baseline (plain looped, sequential T*=n)
  right = c=2 log-budget (efficient, sub-linear)

Each heatmap = linear-probe accuracy of the running product g_1..g_i at
(loop k, position i).  The white markers trace solve_loop[i] (earliest loop the
prefix at position i is decodable, acc >= 0.9); the dashed line is the
sequential y=x reference.  Baseline rides the diagonal (position i computed at
loop i); c=2's wavefront is compressed and plateaus (the associative signature).

Usage:  python scripts/plot_wavefront.py
Writes: figures/wavefront_s5.{png,pdf}
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [
    ("baseline (plain looped): sequential", "runs/s5_curr32_5m/probe_n16.json"),
    ("c=2 log-budget: efficient", "runs/s5_logc2_curr32/probe_n16.json"),
]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    im = None
    for ax, (title, path) in zip(axes, PANELS):
        o = json.load(open(path))
        acc = np.array(o["wavefront_acc"], float)      # [T+1, n+1], col 0 = BOS(nan)
        n, T = o["n"], o["T"]
        M = acc[:, 1:n + 1]                             # positions 1..n
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis",
                       vmin=0, vmax=1,
                       extent=[0.5, n + 0.5, -0.5, T + 0.5])
        # solve_loop wavefront (white markers) + sequential y=x reference
        sl = o["solve_loop"]
        xs = [int(i) for i in sl if sl[i] is not None]
        ys = [sl[i] for i in sl if sl[i] is not None]
        ax.plot(xs, ys, "o-", color="white", ms=4, lw=1.4,
                label="solve loop (acc$\\geq$.9)")
        ax.plot([1, n], [1, n], "--", color="#ff5555", lw=1.3,
                label="sequential $y{=}x$")
        v = o["verdict"]
        ax.set_title(f"{title}\nslope={v['slope_vs_pos']:.2f}  "
                     f"(corr$_{{lin}}$={v['corr_linear']:.2f}, "
                     f"corr$_{{log}}$={v['corr_log2']:.2f})", fontsize=9.5)
        ax.set_xlabel("sequence position $i$")
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85)
    axes[0].set_ylabel("recurrence loop $k$")
    cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cb.set_label("prefix-state decode accuracy", fontsize=9)
    fig.suptitle("Prefix-state wavefront on S5 ($n{=}16$): the log budget "
                 "compresses the sequential diagonal into an efficient wavefront",
                 fontsize=11, y=1.13)
    os.makedirs("figures", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/wavefront_s5.{ext}", dpi=200, bbox_inches="tight")
    print("wrote figures/wavefront_s5.png/.pdf")


if __name__ == "__main__":
    main()
