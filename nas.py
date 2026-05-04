"""
nas.py  —  Hardware-aware NAS for TinyissimoYOLO
=================================================
Combines the original NAS with:
  - Multi-objective Pareto scoring (mAP + RAM + Flash)
  - Invalid architecture rejection before proxy training
  - Diversity preservation via population aging
  - Adaptive mutation rate (explore early, fine-tune late)
  - Stem weight inheritance for warmer proxy starts
  - Per-evaluation CSV logging
  - Per-generation JSON checkpoints
  - Full resume support from any checkpoint

Usage
-----
  # Fresh search
  python nas.py --data person.yaml --ram_limit 500000 --flash_limit 600000

  # Resume from results JSON
  python nas.py --data person.yaml --resume_from nas_results.json

  # Resume from a specific generation checkpoint
  python nas.py --data person.yaml --resume_from nas_runs/checkpoint_gen03.json

  # Analyse results table
  python nas.py --mode analyse --results nas_results.json

  # Fully train the best found architecture
  python nas.py --mode train_best --arch_json best_arch.json --data person.yaml
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

import custom_modules  # noqa — registers DSInception, DSResBlock, CBAM etc.
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv, RepConv, GhostConv, Focus

os.environ["WANDB_MODE"] = "disabled"


# ============================================================
# Search space
# ============================================================

STEM_OPS = [
    ("Conv",      [3, 1],  "standard 3×3"),
    ("DSConv",    [3, 1],  "depthwise-sep"),
    ("GhostConv", [3, 1],  "ghost features"),
    ("Focus",     [3, 1],  "space-to-depth YOLOv5 stem"),
    ("RepConv",   [3, 1],  "re-param block"),
]

STAGE_OPS = [
    ("Conv",       [3, 1], "standard"),
    ("DSConv",     [3, 1], "depthwise-sep"),
    ("C2f",        [True], "CSP bottleneck shortcut=True"),
    ("C3",         [True], "YOLOv5 CSP"),
    ("GhostConv",  [3, 1], "ghost module"),
    ("DSResBlock", [],     "DS residual identity skip"),
    ("RepConv",    [3, 1], "re-param"),
    ("DSInception",[0.5],  "inception module"),
]

STAGE_CHANNELS = {
    "block1": [8, 12, 16, 24],
    "block2": [12, 16, 24],
    "block3": [24, 32, 40],
    "block4": [48, 64, 80],
    "block5": [96, 128, 160],
}

N_REPEATS  = [1, 2, 3]
N_AWARE_N  = [3, 6, 9]

INCEPTION_SPLITS = [
    [16, 16, 16, 16],
    [8,  24, 24,  8],
    [4,  28, 28,  4],
    [24, 16, 16,  8],
]

SEARCH_SPACE = {
    "stem_op_idx":      list(range(len(STEM_OPS))),
    "stem_ch":          STAGE_CHANNELS["block1"],
    "b2_op_idx":        list(range(len(STAGE_OPS[:6]))),
    "b2_ch":            STAGE_CHANNELS["block2"],
    "b2_n":             N_REPEATS,
    "b2_n_aware":       N_AWARE_N,
    "b2_skip":          [True, False],
    "b2_inc_split_idx": list(range(len(INCEPTION_SPLITS))),
    "b3_op_idx":        list(range(len(STAGE_OPS[:6]))),
    "b3_ch":            STAGE_CHANNELS["block3"],
    "b3_n":             N_REPEATS,
    "b3_n_aware":       N_AWARE_N,
    "b3_skip":          [True, False],
    "b3_inc_split_idx": list(range(len(INCEPTION_SPLITS))),
    "b4_op_idx":        list(range(len(STAGE_OPS))),
    "b4_ch":            STAGE_CHANNELS["block4"],
    "b4_n":             N_REPEATS,
    "b4_n_aware":       N_AWARE_N,
    "b4_skip":          [True, False],
    "b4_inc_split_idx": list(range(len(INCEPTION_SPLITS))),
    "b5_op_idx":        list(range(len(STAGE_OPS))),
    "b5_ch":            STAGE_CHANNELS["block5"],
    "b5_n":             N_REPEATS,
    "b5_n_aware":       N_AWARE_N,
    "b5_skip":          [True, False],
    "b5_inc_split_idx": list(range(len(INCEPTION_SPLITS))),
    "b5_final_op_idx":  [None] + list(range(len(STAGE_OPS[:2]))),
    "neck_ch":          [16, 24, 32, 48],
    "activation":       ["ReLU", "ReLU6", "SiLU", "Hardswish"],
}

N_AWARE_MODULES  = {"C2f", "C3"}
FOCUS_INPUT_CH   = 12


# ============================================================
# Architecture dataclass
# ============================================================

@dataclass
class Architecture:
    stem_op_idx:      int            = 0
    stem_ch:          int            = 16
    b2_op_idx:        int            = 0
    b2_ch:            int            = 16
    b2_n:             int            = 1
    b2_n_aware:       int            = 3
    b2_skip:          bool           = True
    b2_inc_split_idx: int            = 1
    b3_op_idx:        int            = 0
    b3_ch:            int            = 32
    b3_n:             int            = 1
    b3_n_aware:       int            = 3
    b3_skip:          bool           = True
    b3_inc_split_idx: int            = 1
    b4_op_idx:        int            = 7
    b4_ch:            int            = 64
    b4_n:             int            = 2
    b4_n_aware:       int            = 3
    b4_skip:          bool           = True
    b4_inc_split_idx: int            = 1
    b5_op_idx:        int            = 7
    b5_ch:            int            = 128
    b5_n:             int            = 2
    b5_n_aware:       int            = 3
    b5_skip:          bool           = False
    b5_inc_split_idx: int            = 1
    b5_final_op_idx:  Optional[int]  = 1
    neck_ch:          int            = 24
    activation:       str            = "ReLU"
    proxy_map50:      float          = 0.0
    est_ram:          int            = 0
    est_flash:        int            = 0
    generation:       int            = 0
    eval_id:          str            = ""
    parameters:       int            = 0
    age:              int            = 0     # generations this candidate has survived

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Architecture":
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})

    @classmethod
    def random(cls) -> "Architecture":
        return cls(**{k: random.choice(v) for k, v in SEARCH_SPACE.items()})

    def stem_op(self):  return STEM_OPS[self.stem_op_idx]
    def b2_op(self):    return STAGE_OPS[self.b2_op_idx]
    def b3_op(self):    return STAGE_OPS[self.b3_op_idx]
    def b4_op(self):    return STAGE_OPS[self.b4_op_idx]
    def b5_op(self):    return STAGE_OPS[self.b5_op_idx]

    def b5_final(self):
        if self.b5_final_op_idx is None:
            return None
        return STAGE_OPS[self.b5_final_op_idx]


# ============================================================
# Multi-objective scoring
# ============================================================

def pareto_score(arch: Architecture,
                 ram_limit: int,
                 flash_limit: int,
                 ram_weight:   float = 0.3,
                 flash_weight: float = 0.1) -> float:
    """
    Combined score: high mAP AND low hardware cost.
    Both fractions normalised to their limits so they are
    comparable to mAP which is in [0, 1].
    """
    mAP        = arch.proxy_map50
    ram_frac   = arch.est_ram   / max(ram_limit,   1)
    flash_frac = arch.est_flash / max(flash_limit, 1)
    penalty    = ram_weight * ram_frac + flash_weight * flash_frac
    return mAP * (1.0 - penalty)


def is_dominated(a: Architecture,
                 population: list[Architecture]) -> bool:
    """True if any other candidate weakly dominates a on all objectives."""
    for b in population:
        if b.eval_id == a.eval_id:
            continue
        if (b.proxy_map50 >= a.proxy_map50 and
                b.est_ram   <= a.est_ram and
                b.est_flash <= a.est_flash and
                (b.proxy_map50 > a.proxy_map50 or b.est_ram < a.est_ram)):
            return True
    return False


def pareto_front(population: list[Architecture]) -> list[Architecture]:
    return [a for a in population if not is_dominated(a, population)]


# ============================================================
# Validity checker
# ============================================================

def is_valid_architecture(arch: Architecture) -> tuple[bool, str]:
    """
    Check structural validity before spending proxy epochs.
    Returns (is_valid, reason_if_invalid).
    """
    # DSResBlock is identity — input and output channels must match.
    for block_name, block_ch, prev_ch, label in [
        (arch.b2_op()[0], arch.b2_ch, arch.stem_ch, "B2"),
        (arch.b3_op()[0], arch.b3_ch, arch.b2_ch,   "B3"),
        (arch.b4_op()[0], arch.b4_ch, arch.b3_ch,   "B4"),
        (arch.b5_op()[0], arch.b5_ch, arch.b4_ch,   "B5"),
    ]:
        if block_name == "DSResBlock" and block_ch != prev_ch:
            return False, (f"{label}: DSResBlock c_in≠c_out "
                           f"({prev_ch}≠{block_ch})")

    # Focus stem needs at least 8 output channels.
    if arch.stem_op()[0] == "Focus" and arch.stem_ch < 8:
        return False, "Focus stem needs stem_ch>=8"

    # C2f/C3 bottleneck rounds to 0 if channels are too narrow.
    for block_name, block_ch, label in [
        (arch.b2_op()[0], arch.b2_ch, "B2"),
        (arch.b3_op()[0], arch.b3_ch, "B3"),
        (arch.b4_op()[0], arch.b4_ch, "B4"),
        (arch.b5_op()[0], arch.b5_ch, "B5"),
    ]:
        if block_name in ("C2f", "C3") and block_ch < 16:
            return False, f"{label}: C2f/C3 needs ch>=16, got {block_ch}"

    # Neck wider than backbone output is wasteful, not wrong, but filter anyway.
    if arch.neck_ch > arch.b5_ch:
        return False, f"neck_ch ({arch.neck_ch}) > b5_ch ({arch.b5_ch})"

    return True, ""


# ============================================================
# Hardware cost estimator
# ============================================================

def estimate_hardware_cost(arch: Architecture,
                           imgsz: int = 128,
                           nc: int = 1) -> tuple[int, int]:

    def conv_p(ci, co, k=3, g=1):
        return (co * (ci // g) * k * k) + (2 * co)

    def ds_p(ci, co, k=3):
        return conv_p(ci, ci, k, g=ci) + conv_p(ci, co, 1)

    def ghost_p(ci, co, k=3):
        half = co // 2
        return conv_p(ci, half, k) + conv_p(half, half, 5, g=half)

    def c2f_p(ci, co, n=1, e=0.5):
        c = max(1, int(co * e))
        return (conv_p(ci, 2*c, 1)
                + n * (conv_p(c, c, 3) + conv_p(c, c, 3))
                + conv_p((2+n)*c, co, 1))

    def c3_p(ci, co, n=1, e=0.5):
        c = max(1, int(co * e))
        return (conv_p(ci, c, 1) * 2
                + n * (conv_p(c, c, 1) + conv_p(c, c, 3))
                + conv_p(2*c, co, 1))

    def inception_p(ci, co, split_idx=0, bn=0.5):
        spl = INCEPTION_SPLITS[split_idx]
        ratio = co / sum(spl)
        sc = [max(4, int(s*ratio)) for s in spl]
        sc[-1] = co - sum(sc[:-1])
        b1, b2, b3, b4 = sc
        m2 = max(4, int(b2*bn)); m3 = max(4, int(b3*bn))
        return (conv_p(ci,b1,1)
                + conv_p(ci,m2,1) + ds_p(m2,b2)
                + conv_p(ci,m3,1) + ds_p(m3,m3) + ds_p(m3,b3)
                + conv_p(ci,b4,1))

    def detect_p(nc, ch):
        reg_max = 4
        c2 = max(16, ch[0]//4, reg_max*4)
        c3 = max(ch[0], min(nc, 100))
        def db(ci, co): return conv_p(ci,co,3)+conv_p(co,co,3)+(co*4*reg_max+4*reg_max)
        def cb(ci, co): return conv_p(ci,co,3)+conv_p(co,co,3)+(co*nc+nc)
        return db(ch[0],c2) + cb(ch[0],c3)

    def mod_p(name, ci, co, n=1, split_idx=0, ib=0.5):
        if name == "Conv":       return conv_p(ci,co) * n
        if name == "DSConv":     return ds_p(ci,co) * n
        if name == "DSResBlock": return (ds_p(ci,ci,3)*2 + 2*ci) * n
        if name == "C2f":        return c2f_p(ci,co,n)
        if name == "C3":         return c3_p(ci,co,n)
        if name == "GhostConv":  return ghost_p(ci,co) * n
        if name == "RepConv":    return (conv_p(ci,co,3)+conv_p(ci,co,1)) * n
        if name == "Focus":      return conv_p(ci*4,co,3)
        if name == "DSInception":return inception_p(ci,co,split_idx,ib) * n
        return conv_p(ci,co,3) * n

    # Flash
    flash = 0
    c = 3
    sn, _, _ = arch.stem_op()
    flash += mod_p(sn, c, arch.stem_ch)
    c = arch.stem_ch

    for (bn,ba,_), bch, bn_r, bn_a, bskip, sidx in [
        (arch.b2_op(), arch.b2_ch, arch.b2_n, arch.b2_n_aware, arch.b2_skip, arch.b2_inc_split_idx),
        (arch.b3_op(), arch.b3_ch, arch.b3_n, arch.b3_n_aware, arch.b3_skip, arch.b3_inc_split_idx),
        (arch.b4_op(), arch.b4_ch, arch.b4_n, arch.b4_n_aware, arch.b4_skip, arch.b4_inc_split_idx),
        (arch.b5_op(), arch.b5_ch, arch.b5_n, arch.b5_n_aware, arch.b5_skip, arch.b5_inc_split_idx),
    ]:
        eff_n = bn_a if bn in N_AWARE_MODULES else bn_r
        ib    = float(ba[0]) if (bn == "DSInception" and ba) else 0.5
        flash += mod_p(bn, c, bch, eff_n, sidx, ib)
        if bskip and c != bch:
            flash += conv_p(c, bch, 1)
        c = bch

    fin = arch.b5_final()
    if fin: flash += mod_p(fin[0], c, c)
    flash += conv_p(c, arch.neck_ch, 1)
    flash += detect_p(nc, [arch.neck_ch])
    flash  = int(flash * 1.05)

    # RAM (peak activation int8)
    h = imgsz
    peak = 0

    def act(ch, h): return ch * h * h

    def fwd_ram(ci, co, h, op="Conv", n=1):
        inp = act(ci, h); out = act(co, h); tmp = 0
        if op in ("C2f","C3"):    tmp = act(max(1,co//2), h) * (2+n)
        elif op == "DSInception": tmp = out
        elif op == "DSResBlock":  tmp = inp
        return inp + out + tmp

    sn, _, _ = arch.stem_op()
    peak = max(peak, fwd_ram(3, arch.stem_ch, h, sn))
    h //= 2; c = arch.stem_ch

    for bn, bch, bn_r, bn_a, bskip, pc, cur_h in [
        (arch.b2_op()[0], arch.b2_ch, arch.b2_n, arch.b2_n_aware, arch.b2_skip, arch.stem_ch, h),
        (arch.b3_op()[0], arch.b3_ch, arch.b3_n, arch.b3_n_aware, arch.b3_skip, arch.b2_ch,   h//2),
        (arch.b4_op()[0], arch.b4_ch, arch.b4_n, arch.b4_n_aware, arch.b4_skip, arch.b3_ch,   h//4),
        (arch.b5_op()[0], arch.b5_ch, arch.b5_n, arch.b5_n_aware, arch.b5_skip, arch.b4_ch,   h//8),
    ]:
        en = bn_a if bn in N_AWARE_MODULES else bn_r
        peak = max(peak, fwd_ram(pc, bch, cur_h, bn, en))
        if bskip:
            peak = max(peak, act(pc,cur_h)*2 + act(bch,cur_h))
        c = bch

    h //= 16
    if fin: peak = max(peak, fwd_ram(c, c, max(h,1), fin[0]))
    peak = max(peak, act(arch.neck_ch, max(h,1))*2)
    ram  = int(peak * 1.5)

    return ram, flash * 4


# ============================================================
# Constraint validator (combines validity + hardware)
# ============================================================

def passes_constraints(arch: Architecture,
                       ram_limit: int,
                       flash_limit: int,
                       imgsz: int,
                       nc: int = 1) -> bool:
    valid, _ = is_valid_architecture(arch)
    if not valid:
        return False
    arch.est_ram, arch.est_flash = estimate_hardware_cost(arch, imgsz, nc)
    return arch.est_ram <= ram_limit and arch.est_flash <= flash_limit


# ============================================================
# Diversity metrics
# ============================================================

def measure_diversity(population: list[Architecture]) -> float:
    if len(population) <= 1:
        return 1.0
    total_u = total_p = 0
    for key, choices in SEARCH_SPACE.items():
        vals = [str(getattr(a, key)) for a in population]
        total_u += len(set(vals))
        total_p += len(choices)
    return total_u / max(total_p, 1)


def adaptive_mutation_rate(gen: int,
                           total_gens: int,
                           diversity: float) -> float:
    base  = 0.4 - 0.3 * gen / max(total_gens, 1)
    boost = max(0.0, 0.3 - diversity) * 0.5
    return min(0.5, base + boost)


# ============================================================
# Stem weight inheritance cache
# ============================================================

_STEM_CACHE: dict[str, dict] = {}


def _save_stem_weights(arch: Architecture, project: str) -> None:
    best_pt = os.path.join(project, f"nas_{arch.eval_id}",
                           "weights", "best.pt")
    if not os.path.exists(best_pt):
        return
    try:
        ckpt  = torch.load(best_pt, map_location="cpu")
        state = ckpt.get("model", ckpt)
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        stem_w = {k: v for k, v in state.items()
                  if k.startswith("model.0.")}
        if stem_w:
            _STEM_CACHE[arch.stem_op()[0]] = stem_w
    except Exception:
        pass


def _apply_stem_weights(model, arch: Architecture) -> None:
    key = arch.stem_op()[0]
    if key not in _STEM_CACHE:
        return
    try:
        cached  = _STEM_CACHE[key]
        current = model.model.state_dict()
        compat  = {k: v for k, v in cached.items()
                   if k in current and current[k].shape == v.shape}
        if compat:
            current.update(compat)
            model.model.load_state_dict(current, strict=False)
    except Exception:
        pass


# ============================================================
# CSV logger
# ============================================================

_LOG_PATH = "nas_log.csv"


def _init_log(path: str = _LOG_PATH) -> None:
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow([
            "timestamp","eval_id","generation","proxy_map50",
            "est_ram_kb","est_flash_kb","pareto_score",
            "stem","b4","b5","activation","parameters",
        ])


def _append_log(arch: Architecture,
                ram_limit: int,
                flash_limit: int,
                path: str = _LOG_PATH) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(),
            arch.eval_id, arch.generation,
            f"{arch.proxy_map50:.5f}",
            arch.est_ram   // 1024,
            arch.est_flash // 1024,
            f"{pareto_score(arch, ram_limit, flash_limit):.5f}",
            arch.stem_op()[0],
            arch.b4_op()[0],
            arch.b5_op()[0],
            arch.activation,
            arch.parameters,
        ])


# ============================================================
# YAML generator  (unchanged from your version)
# ============================================================

def _module_yaml_line(from_idx, n, module_name, extra_args,
                      c1, c2, comment="") -> tuple[str, int]:
    if module_name == "Focus":
        args, out_ch = [c2] + extra_args, c2
    elif module_name == "DSResBlock":
        args, out_ch = [], c1
    elif module_name == "DSInception":
        args, out_ch = extra_args, c2
    elif module_name in N_AWARE_MODULES:
        args, out_ch = [c2] + extra_args, c2
    else:
        args, out_ch = [c2] + extra_args, c2

    fs = (str(from_idx) if not isinstance(from_idx, list)
          else "[" + ", ".join(str(x) for x in from_idx) + "]")
    line = (f"  - [{fs}, {n}, {module_name},"
            f" [{', '.join(str(a) for a in args)}]]"
            + (f"  # {comment}" if comment else ""))
    return line, out_ch


def arch_to_yaml(arch: Architecture, nc: int = 1) -> str:
    lines = [f"# NAS arch  id={arch.eval_id}", f"nc: {nc}", "", "backbone:"]
    layer_idx = 0
    ch_track  = []

    def scaled_inc(total, split_idx, bn=0.5):
        spl   = INCEPTION_SPLITS[split_idx]
        ratio = total / sum(spl)
        sc    = [max(4, int(s*ratio)) for s in spl]
        sc[-1]= total - sum(sc[:-1])
        return [*sc, bn]

    def emit(fi, n, mod, extra, c1, c2, cmt=""):
        nonlocal layer_idx
        ln, oc = _module_yaml_line(fi, n, mod, extra, c1, c2, cmt)
        lines.append(ln); ch_track.append(oc)
        i = layer_idx; layer_idx += 1
        return i, oc

    def pool(cmt=""):
        nonlocal layer_idx
        lines.append(f"  - [-1, 1, nn.MaxPool2d, [2, 2, 0]]  # {cmt}")
        ch_track.append(ch_track[-1])
        i = layer_idx; layer_idx += 1
        return i, ch_track[-1]

    def add_skip(ml, sl, mc, sc, cmt=""):
        nonlocal layer_idx
        if mc == sc:
            lines.append(f"  - [[{ml}, {sl}], 1, Add, []]  # {cmt}")
            ch_track.append(mc); i = layer_idx; layer_idx += 1
            return i, mc
        pi, _ = emit(sl, 1, "Conv", [1,1], sc, mc,
                     f"skip proj {sc}->{mc}")
        lines.append(f"  - [[{ml}, {pi}], 1, Add, []]  # {cmt}")
        ch_track.append(mc); i = layer_idx; layer_idx += 1
        return i, mc

    def build_block(b_name, b_args, b_ch, b_n, b_n_aware, b_skip,
                    b_inc_split, c_prev, prev_pool, label):
        nonlocal layer_idx
        prev, cp = prev_pool, c_prev
        if b_name == "DSInception":
            for i in range(b_n):
                prev, cp = emit(-1,1,"DSInception",
                                scaled_inc(b_ch,b_inc_split),
                                cp, b_ch, f"{label} inc {i+1}/{b_n}")
        elif b_name in N_AWARE_MODULES:
            prev, cp = emit(-1, b_n_aware, b_name, b_args,
                            cp, b_ch, f"{label} {b_name} n={b_n_aware}")
        else:
            for i in range(b_n):
                prev, cp = emit(-1,1,b_name,b_args,cp,b_ch,
                                f"{label}-{i+1} {b_name}->{b_ch}ch")
        if b_skip:
            prev, cp = add_skip(prev, prev_pool, cp, c_prev,
                                f"skip to {label}")
        return prev, cp

    # Stem
    sn, sa, _ = arch.stem_op()
    sc1 = FOCUS_INPUT_CH if sn=="Focus" else 3
    l0, co = emit(-1,1,sn,sa,sc1,arch.stem_ch,
                  f"stem {sn} 3->{arch.stem_ch}")
    lp1, cp1 = pool("128->64")

    # B2
    lb2,cb2 = build_block(
        arch.b2_op()[0], arch.b2_op()[1], arch.b2_ch,
        arch.b2_n, arch.b2_n_aware, arch.b2_skip,
        arch.b2_inc_split_idx, cp1, lp1, "B2")
    lp2, cp2 = pool("64->32")

    # B3
    lb3,cb3 = build_block(
        arch.b3_op()[0], arch.b3_op()[1], arch.b3_ch,
        arch.b3_n, arch.b3_n_aware, arch.b3_skip,
        arch.b3_inc_split_idx, cp2, lp2, "B3")
    lp3, cp3 = pool("32->16")

    # B4
    lb4,cb4 = build_block(
        arch.b4_op()[0], arch.b4_op()[1], arch.b4_ch,
        arch.b4_n, arch.b4_n_aware, arch.b4_skip,
        arch.b4_inc_split_idx, cp3, lp3, "B4")
    lp4, cp4 = pool("16->8")

    # B5
    lb5,cb5 = build_block(
        arch.b5_op()[0], arch.b5_op()[1], arch.b5_ch,
        arch.b5_n, arch.b5_n_aware, arch.b5_skip,
        arch.b5_inc_split_idx, cp4, lp4, "B5")
    lp5, cp5 = pool("8->4")

    # Final refinement
    fin = arch.b5_final()
    if fin:
        lp5, cp5 = emit(-1,1,fin[0],fin[1],cp5,cp5,"final refinement")

    lines += ["", "head:",
              f"  - [-1, 1, Conv, [{arch.neck_ch}, 1, 1]]"
              f"  # neck {cp5}->{arch.neck_ch}ch",
              f"  - [[-1], 1, Detect, [nc]]  # detect 4×4"]
    return "\n".join(lines)


# ============================================================
# Evaluator
# ============================================================

def _extract_map50(model, project: str, eval_id: str, results) -> float:
    """Try four methods in descending reliability."""
    # 1. results.results_dict
    try:
        metrics = getattr(results, "results_dict", {}) if results is not None else {}
        print(f"    Evaluated mAP50(B): {metrics.get('metrics/mAP50(B)'):.4f}")
        map50 = metrics.get("metrics/mAP50(B)")
        return float(map50) if map50 is not None else 0.0
    except Exception:
        pass

    # 2. results.csv with robust parsing
    try:
        csv_p = os.path.join(project, f"nas_{eval_id}", "results.csv")
        if os.path.exists(csv_p):
            df = pd.read_csv(csv_p)
            df.columns = df.columns.str.strip()
            df = df.dropna(how="all")
            cols = [c for c in df.columns if "mAP50" in c and "95" not in c]
            if cols and len(df) > 0:
                s = df[cols[0]].dropna()
                if len(s) > 0:
                    return float(s.iloc[-1])
    except Exception:
        pass

    # 3. Re-validate best.pt
    try:
        best_pt = os.path.join(project, f"nas_{eval_id}",
                               "weights", "best.pt")
        if os.path.exists(best_pt):
            vm = YOLO(best_pt)
            vr = vm.val(data=model.trainer.args.data,
                        imgsz=model.trainer.args.imgsz,
                        verbose=False, plots=False)
            return float(vr.box.map50)
    except Exception:
        pass

    # 4. validator metrics
    try:
        return float(model.trainer.validator.metrics.box.map50)
    except Exception:
        pass

    return 0.0


def evaluate_candidate(arch: Architecture,
                       data_yaml: str,
                       proxy_epochs: int,
                       imgsz: int,
                       device: str,
                       project: str,
                       nc: int = 1) -> float:
    act_map = {
        "ReLU":      nn.ReLU(inplace=True),
        "ReLU6":     nn.ReLU6(inplace=True),
        "SiLU":      nn.SiLU(inplace=True),
        "Hardswish": nn.Hardswish(inplace=True),
    }
    Conv.default_act    = act_map.get(arch.activation, nn.ReLU(inplace=True))
    RepConv.default_act = Conv.default_act

    # Check for existing results
    csv_path = os.path.join(project, f"nas_{arch.eval_id}", "results.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            df = df.dropna(how="all")
            cols = [c for c in df.columns if "mAP50" in c and "95" not in c]
            if cols and len(df) > 0:
                val = float(df[cols[0]].dropna().iloc[-1])
                if val > 0:
                    print(f"    (cached from CSV: {val:.4f})")
                    return val
        except Exception:
            pass

    yaml_path = f"/tmp/nas_{arch.eval_id}.yaml"
    with open(yaml_path, "w") as f:
        f.write(arch_to_yaml(arch, nc=nc))

    try:
        model = YOLO(yaml_path, verbose=False)
        arch.parameters = model.info()[1]

        # Warm-start stem weights if available
        _apply_stem_weights(model, arch)

        results = model.train(
            data     = data_yaml,
            imgsz    = imgsz,
            epochs   = proxy_epochs,
            batch    = 64,
            workers  = 3,
            optimizer= "Adam",
            lr0      = 1e-3,
            cos_lr   = True,
            project  = project,
            name     = f"nas_{arch.eval_id}",
            exist_ok = True,
            verbose  = False,
            save     = True,
            device   = device,
            patience = proxy_epochs,
        )

        map50 = _extract_map50(model, project, arch.eval_id, results)

        # Cache stem weights of good candidates
        if map50 > 0.15:
            _save_stem_weights(arch, project)

        return map50

    except Exception as e:
        print(f"\n  [WARN] {arch.eval_id} failed: {e}")
        import traceback; traceback.print_exc()
        return 0.0


# ============================================================
# Evolutionary operators
# ============================================================

def mutate(arch: Architecture, rate: float = 0.25) -> Architecture:
    child = copy.deepcopy(arch)
    child.proxy_map50 = 0.0
    child.age = 0
    for k, choices in SEARCH_SPACE.items():
        if random.random() < rate:
            setattr(child, k, random.choice(choices))
    return child


def crossover(a: Architecture, b: Architecture) -> Architecture:
    child = copy.deepcopy(a)
    child.proxy_map50 = 0.0
    child.age = 0
    for k in SEARCH_SPACE:
        if random.random() < 0.5:
            setattr(child, k, getattr(b, k))
    return child


def tournament(pop: list[Architecture], k: int = 3,
               ram_limit: int = 500_000,
               flash_limit: int = 700_000) -> Architecture:
    """Tournament selection with Pareto score and age penalty."""
    contestants = random.sample(pop, min(k, len(pop)))
    def score(a):
        return pareto_score(a, ram_limit, flash_limit) - 0.005 * a.age
    return max(contestants, key=score)


def sample_valid(n: int, ram_limit: int, flash_limit: int,
                 imgsz: int, nc: int,
                 new_id_fn,
                 max_attempts: int = 300) -> list[Architecture]:
    out = []
    for _ in range(max_attempts):
        if len(out) >= n:
            break
        a = Architecture.random()
        if passes_constraints(a, ram_limit, flash_limit, imgsz, nc):
            a.eval_id = new_id_fn()
            out.append(a)
    return out


# ============================================================
# Checkpoint helpers
# ============================================================

def _save_results(data: list, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _save_gen_checkpoint(population: list[Architecture],
                         gen: int, project: str) -> None:
    path = os.path.join(project, f"checkpoint_gen{gen:02d}.json")
    with open(path, "w") as f:
        json.dump([a.to_dict() for a in population], f, indent=2)
    print(f"  Checkpoint saved → {path}")


def _load_resume(resume_from: str) -> tuple[list, list, int, int]:
    """
    Load a results JSON or generation checkpoint.
    Returns (all_results, population, last_gen, max_counter).
    """
    with open(resume_from) as f:
        data = json.load(f)

    if not data:
        return [], [], 0, 0

    # Detect format: list of Architecture dicts (checkpoint) vs all_results
    # Both are lists of dicts; checkpoint items are from a single generation.
    all_results = data if isinstance(data, list) else []

    last_gen = max(int(r.get("generation", 0)) for r in all_results)
    last_gen_data = [r for r in all_results
                     if int(r.get("generation",0)) == last_gen]
    population = [Architecture.from_dict(r) for r in last_gen_data]

    max_counter = 0
    for r in all_results:
        try:
            max_counter = max(max_counter,
                              int(str(r.get("eval_id","")).split("_c")[-1]))
        except Exception:
            pass

    return all_results, population, last_gen, max_counter


# ============================================================
# Main search
# ============================================================

def nas_search(
    data_yaml:       str,
    ram_limit:       int   = 500_000,
    flash_limit:     int   = 600_000,
    population_size: int   = 30,
    generations:     int   = 6,
    proxy_epochs:    int   = 3,
    imgsz:           int   = 128,
    device:          str   = "0",
    project:         str   = "nas_runs",
    nc:              int   = 1,
    seed:            int   = 42,
    results_file:    str   = "nas_results.json",
    resume_from:     Optional[str] = None,
) -> Architecture:

    random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(project, exist_ok=True)

    counter   = [0]
    gen_idx   = [0]
    all_results: list     = []
    population: list[Architecture] = []
    start_gen = 1
    has_incomplete = False
    completed_partial: list[Architecture] = []

    def new_id():
        counter[0] += 1
        return f"g{gen_idx[0]:02d}_c{counter[0]:03d}"

    def run_eval(arch: Architecture) -> None:
        print(f"  {arch.eval_id:12s}  "
              f"stem={arch.stem_op()[0]:<10s}  "
              f"b4={arch.b4_op()[0]:<12s}  "
              f"b5={arch.b5_op()[0]:<12s}  "
              f"RAM={arch.est_ram//1024:>4d}KB  "
              f"Flash={arch.est_flash//1024:>4d}KB  ...",
              end="", flush=True)
        arch.proxy_map50 = evaluate_candidate(
            arch, data_yaml, proxy_epochs, imgsz, device, project, nc)
        all_results.append(arch.to_dict())
        _save_results(all_results, results_file)
        _append_log(arch, ram_limit, flash_limit)
        print(f"  mAP={arch.proxy_map50:.4f}")

    # ── Resume or fresh start ─────────────────────────────────────────
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from {resume_from}")
        all_results, population, last_gen, max_ctr = _load_resume(resume_from)
        counter[0] = max_ctr
        gen_idx[0] = last_gen
        start_gen  = last_gen + 1

        population.sort(key=lambda a: a.proxy_map50, reverse=True)
        population = population[:population_size]

        # Fill if population is too small
        if len(population) < population_size:
            if last_gen == 0:
                needed = population_size - len(population)
                new_cands = sample_valid(needed, ram_limit, flash_limit,
                                        imgsz, nc, new_id)
                print(f"  Sampling {needed} new candidates to fill population...")
                for a in new_cands:
                    a.generation = last_gen
                    run_eval(a)
                population += new_cands
            else:
                completed_partial = population.copy()
                has_incomplete = True
                prev_gen_data = [r for r in all_results if int(r.get("generation",0)) == last_gen-1]
                population = [Architecture.from_dict(r) for r in prev_gen_data]
                start_gen = last_gen  # re-evaluate current gen candidates
                


        # Always include all-time best (elitism)
        best_ever = max(all_results, key=lambda r: float(r.get("proxy_map50",0)))
        ba = Architecture.from_dict(best_ever)
        if ba.eval_id not in {a.eval_id for a in population}:
            population.append(ba)
            population.sort(key=lambda a: a.proxy_map50, reverse=True)
            population = population[:population_size]



        print(f"  Loaded {len(all_results)} past evals  "
              f"last_gen={last_gen}  resuming gen={start_gen}")
        print(f"  Best: {ba.eval_id}  mAP={ba.proxy_map50:.4f}\n")

        # Initialise CSV log in append mode (don't overwrite history)
        if not os.path.exists(_LOG_PATH):
            _init_log()

    else:
        # Fresh start
        _init_log()
        print(f"\n{'='*70}")
        print(f"NAS — search space size ~{_space_size():,}")
        print(f"  Population={population_size}  Generations={generations}")
        print(f"  RAM≤{ram_limit//1024}KB  Flash≤{flash_limit//1024}KB")
        print(f"{'='*70}\n")

        # Seed with known-good default
        seed_arch = Architecture()
        if passes_constraints(seed_arch, ram_limit, flash_limit, imgsz, nc):
            seed_arch.eval_id = new_id()
            population = [seed_arch]
        else:
            population = []

        population += sample_valid(population_size - len(population),
                                   ram_limit, flash_limit, imgsz, nc, new_id)

        print(f"Generation 0 — {len(population)} candidates")
        for a in population:
            a.generation = 0
            run_eval(a)

        _save_gen_checkpoint(population, 0, project)

    # ── Generational loop ─────────────────────────────────────────────
    for gen in tqdm(range(start_gen, generations+1),
                    desc="NAS Generations", unit="gen"):
        gen_idx[0] = gen

        # Age existing population
        for a in population:
            a.age += 1

        # Prune very old candidates but always keep the best
        best_current = max(population, key=lambda a: a.proxy_map50)
        population   = ([a for a in population if a.age <= 4]
                        + [best_current])
        # Deduplicate
        seen = set()
        deduped = []
        for a in population:
            if a.eval_id not in seen:
                seen.add(a.eval_id); deduped.append(a)
        population = deduped

        diversity = measure_diversity(population)
        mut_rate  = adaptive_mutation_rate(gen, generations, diversity)

        print(f"\n{'─'*70}")
        print(f"Gen {gen}  best={best_current.proxy_map50:.4f} "
              f"({best_current.eval_id})  "
              f"diversity={diversity:.2f}  mut_rate={mut_rate:.3f}")
        print(f"  stem={best_current.stem_op()[0]}  "
              f"b4={best_current.b4_op()[0]}  "
              f"b5={best_current.b5_op()[0]}  "
              f"act={best_current.activation}")
        print(f"{'─'*70}")

        new_pop = [best_current]   # elitism

        # Add incomplete children from resume if any
        if has_incomplete:
            new_pop += completed_partial
            has_incomplete = False
            completed_partial.clear()
            

        attempts = 0
        while len(new_pop) < population_size and attempts < population_size*30:
            attempts += 1
            r = random.random()
            if r < 0.5:
                parent = tournament(population, ram_limit=ram_limit,
                                    flash_limit=flash_limit)
                child  = mutate(parent, rate=mut_rate)
            elif r < 0.75:
                p1 = tournament(population, ram_limit=ram_limit,
                                flash_limit=flash_limit)
                p2 = tournament(population, ram_limit=ram_limit,
                                flash_limit=flash_limit)
                child = crossover(p1, p2)
                child = mutate(child, rate=mut_rate * 0.4)
            else:
                child = Architecture.random()

            if not passes_constraints(child, ram_limit, flash_limit, imgsz, nc):
                continue
            child.eval_id    = new_id()
            child.generation = gen
            new_pop.append(child)

        print(f"  Evaluating {len(new_pop)-1} new candidates...")
        for a in new_pop[1:]:
            if a.proxy_map50 == 0.0:
                run_eval(a)
            else:
                print(f"  Skipping {a.eval_id} (already evaluated  "
                      f"mAP={a.proxy_map50:.4f})")

        population = new_pop
        _save_gen_checkpoint(population, gen, project)

        # Print Pareto front for this generation
        front = pareto_front(population)
        print(f"  Pareto front: {len(front)} non-dominated candidates")
        for fa in sorted(front, key=lambda a: -a.proxy_map50)[:5]:
            print(f"    {fa.eval_id}  mAP={fa.proxy_map50:.4f}  "
                  f"RAM={fa.est_ram//1024}KB  Flash={fa.est_flash//1024}KB  "
                  f"score={pareto_score(fa, ram_limit, flash_limit):.4f}")

    # ── Final best ────────────────────────────────────────────────────
    all_arch = population + [Architecture.from_dict(r) for r in all_results]
    best = max(all_arch,
               key=lambda a: pareto_score(a, ram_limit, flash_limit))

    print(f"\n{'='*70}")
    print(f"NAS complete.  Best by Pareto score: {best.eval_id}")
    print(f"  mAP={best.proxy_map50:.4f}  "
          f"RAM={best.est_ram//1024}KB  Flash={best.est_flash//1024}KB")
    print(f"  stem={best.stem_op()[0]} ch={best.stem_ch}")
    print(f"  b2={best.b2_op()[0]} ch={best.b2_ch} n={best.b2_n}")
    print(f"  b3={best.b3_op()[0]} ch={best.b3_ch} n={best.b3_n}")
    print(f"  b4={best.b4_op()[0]} ch={best.b4_ch} n={best.b4_n}")
    print(f"  b5={best.b5_op()[0]} ch={best.b5_ch} n={best.b5_n}")
    print(f"  neck={best.neck_ch}ch  act={best.activation}")

    best_yaml = f"nas_best_{best.eval_id}.yaml"
    with open(best_yaml, "w") as f:
        f.write(arch_to_yaml(best, nc=nc))

    best_json = f"nas_best_{best.eval_id}.json"
    with open(best_json, "w") as f:
        json.dump(best.to_dict(), f, indent=2)

    print(f"  YAML → {best_yaml}")
    print(f"  JSON → {best_json}")

    return best


def _space_size() -> int:
    s = 1
    for v in SEARCH_SPACE.values(): s *= len(v)
    return s


# ============================================================
# Full training of winner
# ============================================================

def train_best(arch, data_yaml, epochs=300, imgsz=128,
               device="0", project="nas_runs", nc=1):
    act_map = {
        "ReLU":      nn.ReLU(inplace=True),
        "ReLU6":     nn.ReLU6(inplace=True),
        "SiLU":      nn.SiLU(inplace=True),
        "Hardswish": nn.Hardswish(inplace=True),
    }
    Conv.default_act = act_map.get(arch.activation, nn.ReLU(inplace=True))

    yaml_path = f"nas_best_{arch.eval_id}.yaml"
    if not os.path.exists(yaml_path):
        with open(yaml_path, "w") as f:
            f.write(arch_to_yaml(arch, nc=nc))

    model = YOLO(yaml_path)
    model.train(
        data=data_yaml, imgsz=imgsz, epochs=epochs, batch=32,
        optimizer="AdamW", lr0=2e-4, cos_lr=True, mosaic=1.0,
        patience=50, project=project,
        name=f"nas_best_{arch.eval_id}_full", exist_ok=True,
    )


# ============================================================
# Analysis
# ============================================================

def analyse(results_file: str = "nas_results.json",
            top_k: int = 20,
            ram_limit: int = 500_000,
            flash_limit: int = 600_000):
    with open(results_file) as f:
        results = json.load(f)

    for r in results:
        a = Architecture.from_dict(r)
        r["_pareto"] = pareto_score(a, ram_limit, flash_limit)

    results.sort(key=lambda r: r.get("_pareto", 0), reverse=True)

    hdr = (f"{'ID':<14} {'mAP':>6} {'RAM':>6} {'Flash':>6} "
           f"{'Pareto':>7} {'stem':<12} {'b4':<14} {'b5':<14} "
           f"{'act':<10} {'params':>8}")
    print(f"\nTop {top_k} by Pareto score:")
    print(hdr); print("-" * len(hdr))
    for r in results[:top_k]:
        print(f"{r['eval_id']:<14} "
              f"{r['proxy_map50']:>6.4f} "
              f"{r['est_ram']//1024:>5}K "
              f"{r['est_flash']//1024:>5}K "
              f"{r['_pareto']:>7.4f} "
              f"{STEM_OPS[r['stem_op_idx']][0]:<12} "
              f"{STAGE_OPS[r['b4_op_idx']][0]:<14} "
              f"{STAGE_OPS[r['b5_op_idx']][0]:<14} "
              f"{r['activation']:<10} "
              f"{r.get('parameters',0):>8,}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="NAS for TinyissimoYOLO")
    p.add_argument("--data",          default="person.yaml")
    p.add_argument("--ram_limit",     type=int,   default=500_000)
    p.add_argument("--flash_limit",   type=int,   default=600_000)
    p.add_argument("--population",    type=int,   default=30)
    p.add_argument("--generations",   type=int,   default=6)
    p.add_argument("--proxy_epochs",  type=int,   default=3)
    p.add_argument("--imgsz",         type=int,   default=128)
    p.add_argument("--device",        default="0")
    p.add_argument("--project",       default="nas_runs")
    p.add_argument("--nc",            type=int,   default=1)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--results",       default="nas_results.json")
    p.add_argument("--resume_from",   default=None,
                   help="Path to nas_results.json or checkpoint_genXX.json")
    p.add_argument("--mode",
                   choices=["search","train_best","analyse"],
                   default="search")
    p.add_argument("--arch_json",     default=None)
    args = p.parse_args()

    if args.mode == "search":
        best = nas_search(
            data_yaml       = args.data,
            ram_limit       = args.ram_limit,
            flash_limit     = args.flash_limit,
            population_size = args.population,
            generations     = args.generations,
            proxy_epochs    = args.proxy_epochs,
            imgsz           = args.imgsz,
            device          = args.device,
            project         = args.project,
            nc              = args.nc,
            seed            = args.seed,
            results_file    = args.results,
            resume_from     = args.resume_from,
        )

    elif args.mode == "train_best":
        if not args.arch_json:
            raise ValueError("--arch_json required")
        with open(args.arch_json) as f:
            arch = Architecture.from_dict(json.load(f))
        train_best(arch, args.data, imgsz=args.imgsz,
                   device=args.device, project=args.project, nc=args.nc)

    elif args.mode == "analyse":
        analyse(args.results, ram_limit=args.ram_limit,
                flash_limit=args.flash_limit)