"""Build the causal damage-cone table (E2) from damage_cones.json files.

Regenerates Table `tab:cones` in the paper.  Never retype these numbers; run
this and paste, so the table cannot drift from the artifacts.

    python paperB/cone_table.py [--tex]
"""
import glob
import json
import os
import sys

LENGTHS = [8, 16, 24, 32]


def arm(name):
    if "base" in name:
        return (0, "baseline $T{=}n$")
    if "logc1p5" in name:
        return (1, "budget $c{=}1.5$")
    if "logc2p5" in name:
        return (3, "budget $c{=}2.5$")
    if "logc2" in name:
        return (2, "budget $c{=}2$")
    return (9, name)


def load(path):
    with open(path) as fh:
        j = json.load(fh)
    out = {}
    for n, cell in j.get("cones", {}).items():
        v = cell.get("v_causal")
        if v:
            out[int(n)] = v
    return out


def main():
    tex = "--tex" in sys.argv
    rows = []
    for p in sorted(glob.glob("runs_new/*/damage_cones.json")
                    + glob.glob("runs/*/damage_cones.json")):
        name = os.path.basename(os.path.dirname(p))
        vs = load(p)
        if not vs:
            continue
        ks = sorted(vs)
        growth = vs[ks[-1]] / vs[ks[0]] if len(ks) >= 2 and vs[ks[0]] else None
        rows.append((arm(name), name, vs, growth))
    rows.sort(key=lambda r: (r[0][0], r[1]))

    if not tex:
        print("%-24s %-18s %s" % ("run", "arm", "causal v by length"))
        print("-" * 78)
        for (_, label), name, vs, g in rows:
            cells = "  ".join("n=%-2d:%4.2f" % (n, vs[n]) for n in sorted(vs))
            print("%-24s %-18s %s   growth %s" %
                  (name, label.replace("$", "").replace("{=}", "="), cells,
                   ("%.2fx" % g) if g else "-"))
        base = [r for r in rows if r[0][0] == 0]
        bud = [r for r in rows if r[0][0] in (1, 2, 3)]
        grow = [r for r in bud if r[3] and r[3] > 1.15]
        print("-" * 78)
        print("baselines: %d, all flat at v=1.00: %s"
              % (len(base), all(abs(v - 1.0) < 0.02
                                for _, _, vs, _ in base for v in vs.values())))
        print("budget models: %d, of which v grows (>1.15x): %d"
              % (len(bud), len(grow)))
        return

    print(r"\begin{tabular}{l l rrrr r}")
    print(r"\toprule")
    print(r"model & arm & $n{=}8$ & $n{=}16$ & $n{=}24$ & $n{=}32$ & growth \\")
    print(r"\midrule")
    last = None
    for (order, label), name, vs, g in rows:
        if last is not None and order != last:
            print(r"\midrule")
        last = order
        cells = " & ".join(("%.2f" % vs[n]) if n in vs else "--"
                           for n in LENGTHS)
        gs = ("%.2f$\\times$" % g) if g else "--"
        print("\\texttt{%s} & %s & %s & %s \\\\"
              % (name.replace("_", r"\_"), label, cells, gs))
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
