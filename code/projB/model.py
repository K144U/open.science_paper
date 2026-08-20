"""Project B — weight-tied looped transformer (plan §4.4).

One GPT-2-style decoder block applied T times (weight-tied), with the Phase-1
architecture factors as config flags:

  injection    {True, False}   re-add the input embeddings to the state at
                               every loop (Huginn-style input injection) vs
                               only initializing the state with them
  loop_enc     {"none", "sin"} sinusoidal loop-index encoding added to the
                               state each iteration (H-B1 corollary knob)
  norm         {"pre", "sandwich"}  pre-LN block vs Huginn's sandwich block
  film         {True, False}   FiLM loop-index conditioning of the block
                               (intervention 4 — cheapest possible un-tying)
  prelude/coda {0, 1, ...}     non-tied layers before/after the loop
                               (0/0 = plain looped; 1/1 = Huginn-style)

Positions use RoPE (needed for the OOD length sweep n up to 1024 with train
n <= 64; learned absolute positions cannot extrapolate).  Attention is causal,
via F.scaled_dot_product_attention.

forward(tokens, T) -> logits [B, L, vocab]; return_states=True additionally
returns the loop-wise state tensor [T+1, B, L, d] (float32, detached) for the
Phase-2 probes and core.trajectories diagnostics.

Size presets (single tied core layer, prelude=coda=0):
  5m: d=640 h=10   15m: d=1088 h=16   50m: d=2048 h=16
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoopedConfig:
    vocab_size: int = 121
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 0                    # 0 -> 4*d_model
    core_layers: int = 1             # layers inside the tied loop unit
    prelude_layers: int = 0
    coda_layers: int = 0
    injection: bool = True
    loop_enc: str = "none"           # "none" | "sin"
    norm: str = "pre"                # "pre" | "sandwich"
    film: bool = False
    max_loops: int = 4096            # cap for the sin loop encoding table
    dropout: float = 0.0

    def __post_init__(self):
        if self.d_ff == 0:
            self.d_ff = 4 * self.d_model
        assert self.d_model % self.n_heads == 0
        assert self.norm in ("pre", "sandwich")
        assert self.loop_enc in ("none", "sin")


PRESETS = {
    "tiny": dict(d_model=128, n_heads=4),                # local smoke
    "5m": dict(d_model=640, n_heads=10),
    "15m": dict(d_model=1088, n_heads=16),
    "50m": dict(d_model=2048, n_heads=16),
}


# ------------------------------------------------------------------ modules

def _rope_cache(dim: int, max_len: int, device, dtype=torch.float32):
    inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device,
                                        dtype=dtype) / dim))
    t = torch.arange(max_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv)                          # [L, dim/2]
    return torch.cos(freqs), torch.sin(freqs)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """x [B, H, L, hd]; rotate pairs (even, odd)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[: x.shape[2]], sin[: x.shape[2]]          # [L, hd/2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c],
                       dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: LoopedConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, L, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
                   for t in (q, k, v))
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(o.transpose(1, 2).reshape(B, L, d))


class MLP(nn.Module):
    def __init__(self, cfg: LoopedConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    """Decoder block; pre-LN or sandwich-LN (Huginn norm_1..norm_4)."""

    def __init__(self, cfg: LoopedConfig):
        super().__init__()
        self.sandwich = cfg.norm == "sandwich"
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.norm_1 = nn.LayerNorm(cfg.d_model)
        self.norm_3 = nn.LayerNorm(cfg.d_model)
        if self.sandwich:
            self.norm_2 = nn.LayerNorm(cfg.d_model)
            self.norm_4 = nn.LayerNorm(cfg.d_model)

    def forward(self, x, cos, sin):
        h = self.attn(self.norm_1(x), cos, sin)
        x = x + (self.norm_2(h) if self.sandwich else h)
        h = self.mlp(self.norm_3(x))
        return x + (self.norm_4(h) if self.sandwich else h)


class LoopUnit(nn.Module):
    """The tied unit: core_layers blocks + optional FiLM loop conditioning."""

    def __init__(self, cfg: LoopedConfig):
        super().__init__()
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.core_layers))
        self.film = None
        if cfg.film:
            # loop-index embedding -> per-channel (scale, shift)
            self.film = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.SiLU(),
                nn.Linear(cfg.d_model, 2 * cfg.d_model),
            )

    def forward(self, x, cos, sin, k_emb=None):
        if self.film is not None and k_emb is not None:
            scale, shift = self.film(k_emb).chunk(2, dim=-1)
            x = x * (1 + scale) + shift
        for b in self.blocks:
            x = b(x, cos, sin)
        return x


class LoopedTransformer(nn.Module):
    def __init__(self, cfg: LoopedConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.prelude = nn.ModuleList(Block(cfg)
                                     for _ in range(cfg.prelude_layers))
        self.core = LoopUnit(cfg)
        self.coda = nn.ModuleList(Block(cfg) for _ in range(cfg.coda_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.loop_enc == "sin" or cfg.film:
            le = torch.zeros(cfg.max_loops, cfg.d_model)
            pos = torch.arange(cfg.max_loops).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, cfg.d_model, 2).float()
                            * (-math.log(10000.0) / cfg.d_model))
            le[:, 0::2] = torch.sin(pos * div)
            le[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("loop_emb", le, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, tokens: torch.Tensor, T: int,
                return_states: bool = False):
        cfg = self.cfg
        B, L = tokens.shape
        dev = tokens.device
        cos, sin = _rope_cache(cfg.d_model // cfg.n_heads, L, dev)
        e = self.wte(tokens)
        for b in self.prelude:
            e = b(e, cos, sin)

        x = e
        states = [x.detach().float()] if return_states else None
        for k in range(T):
            if cfg.injection and k > 0:
                x = x + e
            if cfg.loop_enc == "sin":
                x = x + self.loop_emb[k].to(x.dtype)
            k_emb = (self.loop_emb[k].to(x.dtype) if self.core.film is not None
                     else None)
            x = self.core(x, cos, sin, k_emb=k_emb)
            if return_states:
                states.append(x.detach().float())

        for b in self.coda:
            x = b(x, cos, sin)
        logits = self.head(self.ln_f(x))
        if return_states:
            return logits, torch.stack(states, dim=0)  # [T+1, B, L, d]
        return logits


def make_model(preset: str, vocab_size: int, **overrides) -> LoopedTransformer:
    kw = dict(PRESETS[preset])
    kw.update(overrides)
    return LoopedTransformer(LoopedConfig(vocab_size=vocab_size, **kw))
