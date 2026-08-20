"""Project B: the seed lottery made visible, wavefront pair at c=3 (LOCAL).

Two probe heatmaps (probe.py, S5, n=16) for two seeds trained under the SAME
slack budget (T = ceil(3*log2 n)): seed 1 found the efficient wavefront
(slope 0.63, all 16 positions decodable by loop 10) while seed 0 laid down the
sequential diagonal and never solved position 16 within the probe's loop range.
Same data, same architecture, same budget: the algorithm the loop learns under
a slack budget is decided by the seed.

Usage:  python scripts/plot_bimodal.py
Writes: figures/bimodal_s5_c3.{png,pdf}
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [
    ("seed 0: sequential scan", "runs/s5_logc3_curr32/probe_n16.json"),
    ("seed 1: efficient wavefront", "runs/s5_logc3_s1/probe_n16.json"),
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
    fig.suptitle("The seed lottery under a slack budget ($c{=}3$, S5, "
                 "$n{=}16$): two seeds, same budget, two different algorithms",
                 fontsize=11, y=1.13)
    os.makedirs("figures", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/bimodal_s5_c3.{ext}", dpi=200,
                    bbox_inches="tight")
    print("wrote figures/bimodal_s5_c3.png/.pdf")


if __name__ == "__main__":
    main()
