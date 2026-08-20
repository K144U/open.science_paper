"""CPU unit tests for projB/model.py.

Run:  python code/projB/test_model.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from projB.model import LoopedConfig, LoopedTransformer, make_model  # noqa: E402

torch.manual_seed(0)


def main():
    checks = []
    V = 121

    # ---- shapes + loop-count flexibility
    m = make_model("tiny", V)
    tok = torch.randint(0, V, (2, 12))
    for T in (1, 4, 17):
        out = m(tok, T)
        checks.append((f"logits shape @T={T}", out.shape == (2, 12, V)))
    logits, states = m(tok, 5, return_states=True)
    checks.append(("states [T+1,B,L,d]",
                   states.shape == (6, 2, 12, m.cfg.d_model)))

    # ---- weight tying: one core layer's params reused every loop; grads flow
    n_core = sum(p.numel() for p in m.core.parameters())
    loss = m(tok, 8).sum()
    loss.backward()
    g = sum((p.grad is not None and p.grad.abs().sum() > 0)
            for p in m.core.parameters())
    checks.append(("grads reach all tied core params",
                   g == sum(1 for _ in m.core.parameters())))

    # ---- causality: future tokens cannot change past logits
    m.zero_grad()
    with torch.no_grad():
        a = m(tok, 4)
        tok2 = tok.clone()
        tok2[:, -1] = (tok2[:, -1] + 1) % V
        b = m(tok2, 4)
    checks.append(("causal: last-token change leaves prefix logits intact",
                   torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)))

    # ---- factor flags actually change the computation
    base_kw = dict(vocab_size=V, d_model=64, n_heads=4)
    with torch.no_grad():
        outs = {}
        for name, kw in [
            ("base", {}),
            ("no-inject", {"injection": False}),
            ("loop-enc", {"loop_enc": "sin"}),
            ("sandwich", {"norm": "sandwich"}),
            ("film", {"film": True}),
            ("huginn", {"prelude_layers": 1, "coda_layers": 1}),
        ]:
            torch.manual_seed(1)
            mm = LoopedTransformer(LoopedConfig(**base_kw, **kw))
            outs[name] = mm(tok, 6)
        for name in ("no-inject", "loop-enc", "sandwich", "film", "huginn"):
            checks.append((f"factor changes output: {name}",
                           not torch.allclose(outs[name], outs["base"],
                                              atol=1e-4)))

    # ---- injection=True is loop-count sensitive even at fixed point... just
    # verify T changes output (recurrence actually iterates)
    with torch.no_grad():
        checks.append(("T changes output",
                       not torch.allclose(m(tok, 2), m(tok, 9), atol=1e-4)))

    # ---- RoPE length extrapolation: longer input than any seen length works
    with torch.no_grad():
        long_tok = torch.randint(0, V, (1, 300))
        out = m(long_tok, 3)
    checks.append(("length-300 forward OK (RoPE)", out.shape == (1, 300, V)))

    # ---- size presets in the right ballpark
    for preset, target in (("5m", 5e6), ("15m", 15e6)):
        n = make_model(preset, V).n_params()
        checks.append((f"{preset} params ~{target:.0e} (got {n/1e6:.1f}M)",
                       0.6 * target < n < 1.6 * target))

    n_pass = sum(ok for _, ok in checks)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"{n_pass}/{len(checks)} passed")
    if n_pass != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
