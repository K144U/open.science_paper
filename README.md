# Budget Pressure Breaks the Constant-Speed Computation Frontier
Code and result artifacts for **"Budget Pressure Breaks the Constant-Speed Computation Frontier in Looped Transformers"**.

Depth-recurrent ("looped") transformers can express the `O(log n)`
parallel-prefix solution to finite-group state tracking, but trained from
scratch they learn the `O(n)` sequential scan instead. This repository contains
everything needed to reproduce that finding, the interventions that fail to
change it, and the one that does.

## Headline results

| | baseline (`T=n` training) | log budget (`T=⌈2·log₂n⌉`) |
|---|---|---|
| loops to solve `n=32` on S₅ | **32** | **12** |
| `T*(n)` slope | 1.000 | 0.309 |
| probe wavefront slope | 1.02 (a literal diagonal) | 0.53 (compressed) |
| accuracy at `T=64` | 1.000 (flat) | 0.723 (declining) |

Three things worth knowing:

- **The scan is exact.** Every 5M seed solves length `n` at exactly `T*=n`,
  with accuracy ≤ .012 for every `T ≤ 31` and 1.000 at `T=32`. A linear probe
  shows position `i`'s prefix becoming decodable at loop ≈ `i`.
- **Un-tying the loop does not help.** Sinusoidal loop-index encoding and
  per-loop FiLM both leave the scan intact on 8/8 trained seeds. This was our
  pre-registered mechanism hypothesis, and it is falsified.
- **Infeasibility is what flips it.** A hard `⌈c·log₂n⌉` training budget
  induces an efficient algorithm, reliably at `c ∈ [1.5, 2.5]` (10/10 seeds at
  `c=1.5`), degrading to a seed lottery at `c=3` (6/10). Training-dynamics
  probes show why: the model always learns the scan *first* and compresses it
  only once the length curriculum outgrows the budget.

## Layout

```
code/projB/     tasks.py, model.py, train.py, eval_loops.py, probe.py + tests
scripts/        figure, verification, and cluster (PBS) scripts
runs/           94 result artifacts across 75 training runs
figures/        the five paper figures (PNG + PDF)
paper/          main.tex, main.pdf, results_summary.txt
```

`runs/<name>/` holds `loops_vs_length.json` (the dense loop-ladder evaluation)
for all 75 runs, `probe_n16.json` for the 9 probed checkpoints, and
`probe_dyn_*.json` for the 10 training-dynamics checkpoints of
`s5_logc2_curr32`.

## Reproducing the paper

Every figure and every number in the manuscript is derived from the JSON
artifacts in `runs/`, so no GPU and no model weights are needed:

```bash
pip install -r requirements.txt

python scripts/plot_wavefront.py            # Figure 1 (centerpiece)
python scripts/plot_csweep.py               # Figure 2
python scripts/plot_reliability_budget.py   # Figure 3
python scripts/plot_dynamics.py             # Figure 4
python scripts/plot_bimodal.py              # Figure 5

python scripts/dump_paperB_numbers.py       # regenerates paper/results_summary.txt
python scripts/redteam_paperB.py            # re-runs the claim verification
```

Run these from the repository root (they resolve `runs/` and `figures/`
relative to the working directory).

## Retraining from scratch

Training and probing do need a GPU. A 5M run takes roughly 10 to 40 minutes on
an idle A100.

```bash
# baseline: T = n loops (learns the sequential scan)
python code/projB/train.py --task s5 --preset 5m --loop-schedule fixed_n \
    --n-max 32 --curriculum-frac 0.6 --steps 40000 --batch 256 \
    --lr 1e-3 --seed 0 --out runs/s5_baseline

# intervention: hard T = ceil(2*log2 n) budget (induces the efficient algorithm)
python code/projB/train.py --task s5 --preset 5m --loop-schedule log_n --log-c 2 \
    --n-max 32 --curriculum-frac 0.6 --steps 40000 --batch 256 \
    --lr 1e-3 --seed 0 --out runs/s5_logc2

# loop ladder -> runs/s5_logc2/loops_vs_length.json
python code/projB/eval_loops.py --task s5 --ckpt runs/s5_logc2/ckpt_final.pt \
    --out runs/s5_logc2/loops_vs_length.json

# wavefront probe -> runs/s5_logc2/probe_n16.json
python code/projB/probe.py --task s5 --ckpt runs/s5_logc2/ckpt_final.pt \
    --n 16 --out runs/s5_logc2/probe_n16.json
```

`scripts/submit_*.sh` and `scripts/run_projB_*.pbs` are the exact cluster job
scripts used for the study, kept for provenance.

## Model weights

Trained weights are **not** in this repository. There are 85 checkpoints
totalling ~2.6 GB, and the four 50M checkpoints are 194 MB each, past GitHub's
100 MB per-file limit. They are not needed to reproduce any figure or number in
the paper, only to re-run `eval_loops.py` / `probe.py` on the existing runs.

Checkpoints are slim and model-only (`{step, cfg, model}`), exactly what
`eval_loops.py` and `probe.py` load. FiLM and loop-index encoding are
architecture flags inside the checkpoint config, not separate adapter files.

## Tests

Both suites are plain scripts (no pytest needed); `test_model.py` runs on CPU.

```bash
python code/projB/test_tasks.py
python code/projB/test_model.py
```

`test_tasks.py` checks the group Cayley tables (associativity, identity,
inverses, S₅ non-solvability) and the held-out transition-pair split;
`test_model.py` checks the looped forward pass, weight tying, and state
collection.

## Citation

```bibtex
@article{anon2026wrongloop,
  title   = {Budget Pressure Breaks the Constant-Speed Computation Frontier in Looped Transformers},
  author  = {Anonymous},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

MIT (see `LICENSE`).
