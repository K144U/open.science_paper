"""Build the paper's headline figure: frontier speed against sequence length.

The main text previously carried one figure, illustrating the sequential scan,
which is the part closest to what concurrent work already showed. Contributions
1, 2 and 4 were conveyed entirely through tables, so a skimming reviewer had to
read Table 1 to see the headline claim. This figure states it directly.

  LEFT   behavioural v(n) = n / T*(n), one line per run, unbudgeted against
         budget-trained. Under a constant-speed frontier every line is flat.
  RIGHT  causal v(n) from activation patching, same contrast. This is the panel
         that excludes an affine frontier, since a one-off start-up cost cannot
         grow.

Populations follow paperB/populations.py exactly, so the figure and Tables 1
and 2 cannot drift apart.

    python paperB/make_frontier_fig.py
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from populations import arm_of           # noqa: E402
from frontier_test import load_curve, MIN_RUNGS   # noqa: E402

UNB = ("baseline", "un-tying")
C_UNB, C_BUD = "#3b6ea5", "#c0392b"


def behavioural_curves():
    unb, bud = [], []
    for d in sorted(glob.glob("runs/s5_*") + glob.glob("runs_new/s5_*")):
        if not os.path.isdir(d):
            continue
        arm = arm_of(os.path.basename(d))
        if arm is None:
            continue
        pts = load_curve(d)
        if not pts or len(pts) < MIN_RUNGS:
            continue
        xy = [(n, n / t) for n, t in pts]
        (unb if arm in UNB else bud).append(xy)
    return unb, bud


def causal_curves():
    unb, bud = [], []
    for p in (glob.glob("runs/*/damage_cones.json") +
              glob.glob("runs_new/*/damage_cones.json")):
        name = os.path.basename(os.path.dirname(p))
        if not name.startswith("s5_"):
            continue
        arm = arm_of(name)
        if arm is None:
            continue
        try:
            j = json.load(open(p))
        except Exception:
            continue
        xy = sorted((int(n), c["v_causal"])
                    for n, c in j.get("cones", {}).items() if c.get("v_causal"))
        if len(xy) < 2:
            continue
        (unb if arm in UNB else bud).append(xy)
    return unb, bud


def panel(ax, unb, bud, title, ylab, logy=False):
    # budget-trained underneath, unbudgeted on top: the unbudgeted runs lie on
    # top of one another at v=1, so if they are drawn first a single red run
    # crossing that region reads as a rising blue line.
    for xy in bud:
        ax.plot([p[0] for p in xy], [p[1] for p in xy],
                color=C_BUD, lw=1.0, alpha=.35, zorder=2)
    for xy in unb:
        ax.plot([p[0] for p in xy], [p[1] for p in xy],
                color=C_UNB, lw=1.4, alpha=.85, zorder=4)
    ax.axhline(1.0, color="0.35", ls=":", lw=1.0, zorder=1)
    ax.set_xlabel("sequence length $n$")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([4, 8, 16, 24, 32])
    if logy:
        # two budgeted checkpoints reach v ~ 60, which on a linear axis
        # compresses every other run into a flat band and hides the effect.
        ax.set_yscale("log")
        ax.set_yticks([1, 2, 5, 10, 20, 50])
        ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.02)


def main():
    bu, bb = behavioural_curves()
    cu, cb = causal_curves()
    print("behavioural: %d unbudgeted, %d budget" % (len(bu), len(bb)))
    print("causal     : %d unbudgeted, %d budget" % (len(cu), len(cb)))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.05))
    panel(axes[0], bu, bb, "Behavioural: $v(n)=n/T^*(n)$", "positions per loop")
    panel(axes[1], cu, cb, "Causal: activation patching", "positions per loop",
          logy=True)

    h = [plt.Line2D([], [], color=C_UNB, lw=1.6),
         plt.Line2D([], [], color=C_BUD, lw=1.6),
         plt.Line2D([], [], color="0.35", ls=":", lw=1.2)]
    axes[1].legend(h, ["unbudgeted", "budget-trained", "constant speed $v{=}1$"],
                   fontsize=7, frameon=False, loc="upper left")

    fig.tight_layout(pad=0.4)
    out = os.path.join(HERE, "figures", "frontier_speed.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
