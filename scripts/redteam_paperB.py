"""Red-team verification of every checkable Paper B claim against raw JSONs."""
import glob
import json
import os

import numpy as np


def curve(path):
    o = json.load(open(os.path.join(path, "loops_vs_length.json")))
    c = o["curve"][0] if isinstance(o["curve"], list) else o["curve"]
    return c


print("== CLAIM: baseline threshold sharp, acc at chance for EVERY T<T* ==")
for r in ["s5_curr32_5m", "s5_base_s1", "s5_base_s2",
          "a5_base_s0", "a5_base_s1", "a5_base_s2"]:
    c = curve(f"runs/{r}")
    worst = []
    for n, d in c.items():
        ts = d["T_star"]
        if ts is None:
            continue
        below = {int(t): a for t, a in d["acc_by_T"].items() if int(t) < ts}
        if below:
            bt = max(below, key=below.get)
            worst.append((int(n), bt, below[bt]))
    mx = max(worst, key=lambda x: x[2])
    print(f"  {r}: worst sub-T* acc = {mx[2]:.4f} at (n={mx[0]}, T={mx[1]}); "
          f"chance={1/120 if 's5' in r else 1/60:.4f}")

print("\n== CLAIM: heldout at T* == iid (baseline + c=2 seed0) ==")
for r in ["s5_curr32_5m", "s5_base_s1", "s5_base_s2", "s5_logc2_curr32",
          "a5_base_s0", "a5_logc2_s0"]:
    c = curve(f"runs/{r}")
    rows = []
    for n, d in sorted(c.items(), key=lambda x: int(x[0])):
        if d["T_star"] is not None:
            iid = d["acc_by_T"][str(d["T_star"])]
            rows.append(f"n{n}:iid{iid:.3f}/ho{d.get('heldout_at_Tstar')}")
    print(f"  {r}: {'  '.join(rows)}")

print("\n== CLAIM: non-reach-32 budget seeds have GENUINE plateaus (acc<=.25 "
      "at n=32, all tested T) ==")
seeds = (["s5_logc1p5_s%d" % s for s in range(10)]
         + ["s5_logc2_curr32"] + ["s5_logc2_s%d" % s for s in range(1, 10)]
         + ["s5_logc2p5_s%d" % s for s in range(10)]
         + ["s5_logc3_curr32"] + ["s5_logc3_s%d" % s for s in range(1, 10)])
bad = []
for r in seeds:
    c = curve(f"runs/{r}")
    solved = [int(n) for n, d in c.items() if d["T_star"] is not None]
    reach = max(solved) if solved else 0
    if reach < 32 and "32" in c:
        best = max(c["32"]["acc_by_T"].values())
        maxT = max(int(t) for t in c["32"]["acc_by_T"])
        flag = "  <-- VIOLATES" if best > 0.25 else ""
        if best > 0.20 or flag:
            print(f"  {r}: reach={reach} best@n32={best:.3f} maxT={maxT}{flag}")
        if best > 0.25:
            bad.append(r)
print(f"  violators: {bad if bad else 'NONE'}")

print("\n== CLAIM: coefficient medians per c (paper says ~2.3?, 2.7, 3.7) ==")
groups = {
    1.5: ["s5_logc1p5_s%d" % s for s in range(10)],
    2.0: ["s5_logc2_curr32"] + ["s5_logc2_s%d" % s for s in range(1, 10)],
    2.5: ["s5_logc2p5_s%d" % s for s in range(10)],
    3.0: ["s5_logc3_curr32"] + ["s5_logc3_s%d" % s for s in range(1, 10)],
}
for cval, rs in groups.items():
    coefs = []
    for r in rs:
        c = curve(f"runs/{r}")
        pts = sorted((int(n), d["T_star"]) for n, d in c.items()
                     if d["T_star"] is not None)
        pos = np.array([n for n, _ in pts], float)
        ts = np.array([t for _, t in pts], float)
        slope = np.polyfit(pos, ts, 1)[0]
        if slope <= 0.6:                       # efficient seeds only
            coefs.append(np.polyfit(np.log2(pos), ts, 1)[0])
    print(f"  c={cval}: n_eff={len(coefs)} median coef="
          f"{np.median(coefs):.2f} range=[{min(coefs):.2f},{max(coefs):.2f}]")

print("\n== CLAIM sensitivity: efficient counts under cutoff .5 vs .6 ==")
for cval, rs in groups.items():
    n5 = n6 = 0
    for r in rs:
        c = curve(f"runs/{r}")
        pts = sorted((int(n), d["T_star"]) for n, d in c.items()
                     if d["T_star"] is not None)
        s = np.polyfit([p for p, _ in pts], [t for _, t in pts], 1)[0]
        n6 += s <= 0.6
        n5 += s <= 0.5
    print(f"  c={cval}: <=0.6 -> {n6}/10   <=0.5 -> {n5}/10")

print("\n== scan feasibility crossover: max n with ceil(c*log2 n) >= n ==")
import math
for cv in (1.5, 2.0, 2.5, 3.0):
    ns = [n for n in range(2, 33) if math.ceil(cv * math.log2(n)) >= n]
    cov32 = math.ceil(cv * math.log2(32)) / 32
    print(f"  c={cv}: scan fully feasible n<={max(ns)}; budget/n at n=32 = "
          f"{cov32:.2f}")

print("\n== counts for reproducibility para ==")
evals = glob.glob("runs/*/loops_vs_length.json")
probes = glob.glob("runs/*/probe_n16.json") + \
    glob.glob("runs/*/probe_dyn_*.json")
print(f"  local eval JSONs: {len(evals)}  probe JSONs: {len(probes)}")
