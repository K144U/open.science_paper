"""Dump every number Paper B cites into paper/results_summary.txt (LOCAL).

Walks runs/*/loops_vs_length.json and runs/*/probe_n16.json (+ probe_dyn_*)
and prints, per run: T*(n) table, linear slope, log2 fit (a*log2 n + b, corr),
reach, and probe verdicts.  Also the c=2 seed-0 "acc peaks then declines"
ladder at n=32 and the 10-seed reliability classification per budget.
"""
import glob
import json
import os
import re

import numpy as np

OUT = "paper/results_summary.txt"
lines = []


def p(s=""):
    lines.append(s)
    print(s)


def curve(path):
    o = json.load(open(os.path.join(path, "loops_vs_length.json")))
    cur = o["curve"][0] if isinstance(o["curve"], list) else o["curve"]
    return o, cur


def stats(path):
    o, cur = curve(path)
    solved = sorted((int(n), d["T_star"]) for n, d in cur.items()
                    if d["T_star"] is not None)
    pos = np.array([n for n, _ in solved], float)
    ts = np.array([t for _, t in solved], float)
    if len(solved) < 2:
        return solved, None, None, (int(pos.max()) if len(solved) else 0)
    sl = float(np.polyfit(pos, ts, 1)[0])
    a, b = np.polyfit(np.log2(pos), ts, 1)
    cl = float(np.corrcoef(pos, ts)[0, 1])
    cg = float(np.corrcoef(np.log2(pos), ts)[0, 1])
    return solved, sl, (float(a), float(b), cl, cg), int(pos.max())


def show(name):
    path = f"runs/{name}"
    if not os.path.exists(os.path.join(path, "loops_vs_length.json")):
        p(f"{name}: MISSING")
        return
    solved, sl, fit, reach = stats(path)
    tstr = " ".join(f"({n},{t})" for n, t in solved)
    if sl is None:
        p(f"{name}: T*={tstr} reach={reach} (too few points to fit)")
        return
    a, b, cl, cg = fit
    p(f"{name}: T*={tstr} slope={sl:.3f} reach={reach} "
      f"logfit={a:.2f}*log2n{b:+.2f} corr_lin={cl:.3f} corr_log={cg:.3f}")


GROUPS = {
    "S5 baseline (fixed T=n)": ["s5_curr32_5m", "s5_base_s1", "s5_base_s2"],
    "S5 loop_enc": ["s5_le_curr32", "s5_le_s1", "s5_le_s2", "s5_le_s3",
                    "s5_le_s4", "s5_le_s5"],
    "S5 film (batch256)": ["s5_film_s0", "s5_film_s1", "s5_film_s2"],
    "S5 film (batch128, superseded)": ["s5_film_curr32"],
    "S5 c=1": ["s5_logc1_curr32", "s5_logc1_s1", "s5_logc1_s2"],
    "S5 c=1.5": [f"s5_logc1p5_s{s}" for s in range(10)],
    "S5 c=2": ["s5_logc2_curr32"] + [f"s5_logc2_s{s}" for s in range(1, 10)],
    "S5 c=2.5": [f"s5_logc2p5_s{s}" for s in range(10)],
    "S5 c=3": ["s5_logc3_curr32"] + [f"s5_logc3_s{s}" for s in range(1, 10)],
    "S5 c=2 n_max=64": ["s5_logc2_n64_s0"],
    "A5 baseline": ["a5_base_s0", "a5_base_s1", "a5_base_s2"],
    "A5 c=2": ["a5_logc2_s0", "a5_logc2_s1", "a5_logc2_s2"],
    "mod60": ["mod60_base_s0", "mod60_logc2_s0"],
    "15m": ["s5_15m_base_s0", "s5_15m_base_s1", "s5_15m_base_s2",
            "s5_15m_logc2_s0", "s5_15m_logc2_s1", "s5_15m_logc2_s2"],
    "50m": ["s5_50m_base_s0", "s5_50m_logc2_s0", "s5_50m_logc2_s1",
            "s5_50m_logc2_s2"],
}

for title, names in GROUPS.items():
    p(f"== {title} ==")
    for n in names:
        show(n)
    p()

p("== smoking gun: c=2 seed0 acc_by_T at n=32 ==")
o, cur = curve("runs/s5_logc2_curr32")
ab = cur["32"]["acc_by_T"]
for t in sorted(ab, key=int):
    p(f"  T={t}: {ab[t]:.3f}")
p()

p("== baseline sharp threshold: seed0 acc_by_T at n=32 (T 28..34, 64) ==")
o, cur = curve("runs/s5_curr32_5m")
ab = cur["32"]["acc_by_T"]
for t in sorted(ab, key=int):
    if 28 <= int(t) <= 34 or int(t) in (16, 24, 64):
        p(f"  T={t}: {ab[t]:.3f}")
p()

p("== probes (probe_n16.json verdicts) ==")
for path in sorted(glob.glob("runs/*/probe_n16.json")):
    o = json.load(open(path))
    v = o["verdict"]
    sl = o["solve_loop"]
    seq = [sl[str(i)] for i in range(1, o["n"] + 1)]
    p(f"{path.split(os.sep)[-2] if os.sep in path else path.split('/')[-2]}: "
      f"slope={v['slope_vs_pos']:.3f} corr_lin={v['corr_linear']:.3f} "
      f"corr_log={v['corr_log2']:.3f} solve_loop={seq}")
    if "block_ladder" in o:
        p(f"   block_ladder={o['block_ladder']}")
p()

p("== dynamics movie (s5_logc2_curr32 probe_dyn_*) ==")
for path in sorted(glob.glob("runs/s5_logc2_curr32/probe_dyn_*.json"),
                   key=lambda x: int(re.search(r"(\d+)\.json", x).group(1))):
    o = json.load(open(path))
    step = int(re.search(r"(\d+)\.json", path).group(1))
    v = o["verdict"]
    k = sum(x is not None for x in o["solve_loop"].values())
    p(f"  step {step:>6}: slope={v.get('slope_vs_pos', float('nan')):.3f} "
      f"solved={k}/16 corr_lin={v.get('corr_linear', float('nan')):.3f} "
      f"corr_log={v.get('corr_log2', float('nan')):.3f}")
p()

p("== 10-seed reliability classification (eff<=.6, seq>=.8) ==")
for c, names in [(1.5, GROUPS["S5 c=1.5"]), (2.0, GROUPS["S5 c=2"]),
                 (2.5, GROUPS["S5 c=2.5"]), (3.0, GROUPS["S5 c=3"])]:
    cls, r32 = [], 0
    for n in names:
        _, sl, _, reach = stats(f"runs/{n}")
        cls.append("eff" if sl <= 0.6 else ("seq" if sl >= 0.8 else "mixed"))
        r32 += reach >= 32
    p(f"  c={c}: eff={cls.count('eff')}/10 seq={cls.count('seq')} "
      f"mixed={cls.count('mixed')} reach32={r32}/10")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n")
print(f"\nwrote {OUT}")
