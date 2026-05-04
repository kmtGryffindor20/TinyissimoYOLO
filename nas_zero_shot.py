"""
nas_zero_shot.py
=================
Training-free NAS for TinyissimoYOLO using zero-cost proxies.

Instead of training each candidate for N proxy epochs, each architecture
is scored using a combination of:

  1. SynFlow    — parameter magnitude product (trainability, no data)
  2. SNIP       — connection sensitivity to loss (trainability, needs 1 batch)
  3. NASWOT     — Hamming distance of activation patterns (expressivity)
  4. Jacobian   — condition number of input-output Jacobian (trainability)
  5. Param/FLOP — analytical efficiency metrics (hardware constraint)

Why these proxies
------------------
SynFlow avoids the layer-collapse problem of earlier pruning proxies.
It scores each architecture without any data by propagating ones through
the network and measuring the product of all weight magnitudes.
Higher SynFlow = weights are better distributed across layers = more
likely to train well without one layer dominating.

SNIP measures which connections matter most for the initial loss gradient.
Networks where removing connections causes large loss changes are networks
with high sensitivity = better signal flow = faster convergence.

NASWOT counts how many distinct activation patterns a network produces
across a batch of inputs. More distinct patterns means the network can
distinguish more inputs = higher expressivity = better potential accuracy.

Jacobian condition number measures whether gradients flow uniformly
through the network. A badly conditioned Jacobian means some directions
vanish and others explode — the network cannot be trained stably.
Lower condition number = better trainability.

How these relate to mAP
------------------------
None of these proxies directly predict mAP. What they predict is
"given random initialisation, how likely is this architecture to learn
good feature representations when trained". The connection to mAP:

  - High expressivity (NASWOT) → can represent complex decision boundaries
                                  → higher potential mAP ceiling
  - Good trainability (SynFlow, Jacobian) → will reach that ceiling faster
                                             → better mAP at convergence
  - High sensitivity (SNIP) → gradients flow to all layers
                               → all layers contribute to features
                               → more diverse features → better mAP

Empirically, the combination of SynFlow + NASWOT + Jacobian achieves
Spearman rank correlation of 0.6-0.8 with final accuracy on standard
NAS benchmarks (NAS-Bench-201, DARTS). This is not perfect but is
sufficient to filter out bad architectures and prioritise good ones
before committing to full training.

Limitation
-----------
These proxies are correlational, not causal. They work better for
ranking architectures within the same search space than for absolute
prediction. Use this script to produce a shortlist of 5-10 candidates,
then use the proxy training NAS (nas.py) to evaluate that shortlist.

Usage
-----
  # Score all architectures in search space (sampled)
  python nas_zero_shot.py --data person.yaml --samples 500

  # Score and then verify top-K with 3-epoch proxy training
  python nas_zero_shot.py --data person.yaml --samples 500 \
      --verify_topk 10 --proxy_epochs 3

  # Analyse results from a previous run
  python nas_zero_shot.py --mode analyse --results zs_results.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
import warnings
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Import your architecture definition and search space
# (copies the relevant parts so this file is standalone)
import custom_modules  # noqa
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv, RepConv
os.environ["WANDB_MODE"] = "disabled"

# Re-use search space and Architecture from nas.py
from nas import (
    Architecture, SEARCH_SPACE, STEM_OPS, STAGE_OPS,
    INCEPTION_SPLITS, N_AWARE_MODULES, FOCUS_INPUT_CH,
    arch_to_yaml, passes_constraints, estimate_hardware_cost, evaluate_candidate,
    pareto_score, _space_size,
)


# ============================================================
# Model builder — builds a PyTorch model from Architecture
# ============================================================

def build_model(arch: Architecture,
                nc: int = 1,
                imgsz: int = 128) -> Optional[nn.Module]:
    """
    Build a Ultralytics model from an Architecture and return the
    underlying nn.Module for proxy computation.
    Returns None if the architecture is invalid.
    """
    try:
        act_map = {
            "ReLU":      nn.ReLU(inplace=True),
            "ReLU6":     nn.ReLU6(inplace=True),
            "SiLU":      nn.SiLU(inplace=True),
            "Hardswish": nn.Hardswish(inplace=True),
        }
        Conv.default_act    = act_map.get(arch.activation, nn.ReLU(inplace=True))
        RepConv.default_act = Conv.default_act

        yaml_path = f"/tmp/zs_nas_{arch.eval_id}.yaml"
        with open(yaml_path, "w") as f:
            f.write(arch_to_yaml(arch, nc=nc))

        model_wrapper = YOLO(yaml_path, verbose=False)
        model         = model_wrapper.model

        # Store parameter count
        arch.parameters = sum(p.numel() for p in model.parameters()
                               if p.requires_grad)
        return model

    except Exception as e:
        return None


def get_dummy_batch(imgsz: int = 128,
                    batch_size: int = 8,
                    device: str = "cpu") -> torch.Tensor:
    """Random input batch for proxy computation."""
    return torch.randn(batch_size, 3, imgsz, imgsz, device=device)


def get_dummy_targets(model: nn.Module,
                      x: torch.Tensor,
                      nc: int = 1) -> dict:
    """
    Build a minimal batch dict for Ultralytics loss computation.
    Used by SNIP/GraSP which need a loss value.
    """
    B = x.size(0)
    # Create fake bounding boxes (one per image, centred)
    batch_idx = torch.arange(B, dtype=torch.float32)
    cls       = torch.zeros(B, 1)
    bboxes    = torch.tensor([[0.5, 0.5, 0.4, 0.4]] * B)
    return {
        "batch_idx": batch_idx,
        "cls":       cls,
        "bboxes":    bboxes,
        "img":       x,
    }


# ============================================================
# Zero-cost proxy implementations
# ============================================================

class ZeroCostProxy:
    """Base class for all zero-cost proxies."""

    name: str = "base"

    def score(self,
              model:   nn.Module,
              x:       torch.Tensor,
              targets: dict = None) -> float:
        raise NotImplementedError

    def __call__(self, model, x, targets=None) -> float:
        try:
            return self.score(model, x, targets)
        except Exception as e:
            warnings.warn(f"{self.name} failed: {e}")
            return 0.0


# ── Proxy 1: SynFlow ─────────────────────────────────────────────────────────

class SynFlow(ZeroCostProxy):
    """
    SynFlow (Tanaka et al. 2020) — Synaptic Flow.

    Computes the product of absolute parameter magnitudes via a single
    forward pass with an all-ones input.  Measures how much of the
    gradient signal each weight carries (its 'synaptic flow').

    Key property: layer-collapse free — does not concentrate all score
    on the widest layer, unlike earlier magnitude-based pruning proxies.

    Score = Σ |θ| × |∂L/∂θ|  summed over all parameters
    where L = Σ(all outputs) and input = ones tensor.

    Higher score → better distributed trainability across all layers.
    No data needed — uses ones input.
    """
    name = "synflow"

    def score(self, model, x, targets=None):
        model.eval()
        # SynFlow uses all-ones input regardless of actual data
        ones = torch.ones_like(x[:1])  # single sample of ones

        # Forward with all-ones, compute synthetic loss = sum of all outputs
        # Use a linearised model (replace activations with identity) is the
        # strict formulation, but the standard approximation is sufficient:
        params = [p for p in model.parameters() if p.requires_grad]

        # Temporarily make all params positive (SynFlow works on magnitude)
        signs = [p.data.sign() for p in params]
        for p in params:
            p.data.abs_()

        # Forward pass
        model.zero_grad()
        try:
            out = model(ones)
            # Handle both tensor and dict outputs
            if isinstance(out, (tuple, list)):
                out = out[0]
            if isinstance(out, dict):
                # Ultralytics detection output during eval — take boxes
                out = list(out.values())[0]
            loss = out.sum()
        except Exception:
            # Restore signs and return 0
            for p, s in zip(params, signs):
                p.data.mul_(s)
            return 0.0

        loss.backward()

        # SynFlow score = sum of |param| * |grad|
        score = 0.0
        for p in params:
            if p.grad is not None:
                score += (p.data * p.grad.data.abs()).sum().item()

        # Restore original signs
        for p, s in zip(params, signs):
            p.data.mul_(s)

        model.zero_grad()
        return float(score)


# ── Proxy 2: SNIP ────────────────────────────────────────────────────────────

class SNIP(ZeroCostProxy):
    """
    SNIP (Lee et al. 2019) — Single-shot Network Pruning based on
    Connection Sensitivity.

    Measures how much the loss changes when each connection is removed.
    Connections with high sensitivity are important for learning.

    Score = Σ |g_i × w_i| / Σ |g_i × w_i| (normalised sum)
    where g_i = ∂L/∂w_i (gradient of loss w.r.t. each weight)

    We aggregate to a single scalar by summing all sensitivity scores.

    Higher score → more connections carry meaningful gradient signal
                  → faster learning → better final accuracy.

    Requires: one forward+backward pass with a few real images.
    """
    name = "snip"

    def score(self, model, x, targets=None):
        model.train()
        model.zero_grad()

        # Forward pass through backbone only (avoid Detect head complexity)
        # Extract intermediate features instead of full detection output
        backbone_out = None
        try:
            # Hook to capture last backbone activation
            hooks = []
            activations = []

            def hook_fn(module, inp, out):
                activations.append(out)

            # Attach hook to the last backbone layer
            backbone_layers = list(model.model.children())
            if backbone_layers:
                h = backbone_layers[-2].register_forward_hook(hook_fn)
                hooks.append(h)

            _ = model(x)
            for h in hooks: h.remove()

            if activations:
                feat = activations[-1]
                if isinstance(feat, (list, tuple)):
                    feat = feat[0]
                loss = feat.abs().mean()
            else:
                return 0.0

        except Exception:
            model.zero_grad()
            return 0.0

        loss.backward()

        score = 0.0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                score += (p.data.abs() * p.grad.data.abs()).sum().item()

        model.zero_grad()
        return float(score)


# ── Proxy 3: NASWOT ──────────────────────────────────────────────────────────

class NASWOT(ZeroCostProxy):
    """
    NASWOT (Mellor et al. 2021) — Neural Architecture Search Without Training.

    Measures network expressivity by counting how many distinct binary
    activation codes the network produces across a batch of inputs.

    For each ReLU, record whether it fires (1) or not (0) for each input.
    Stack these into a binary code per input.
    Compute the Hamming distance matrix between all pairs of codes.
    If many pairs have large Hamming distance → network maps inputs to
    diverse regions → high expressivity → better potential accuracy.

    Score = log|K|  where K is the kernel matrix of binary codes
    (log determinant of the activation kernel matrix, clipped for stability).

    Higher score → more distinct activation patterns → higher expressivity.

    Requires: one forward pass with a batch of diverse inputs.
    No gradients needed.
    """
    name = "naswot"

    def score(self, model, x, targets=None):
        model.eval()

        # Collect binary activation patterns from all ReLU/ReLU6 layers
        activation_codes = []
        hooks = []

        def make_hook():
            def hook_fn(module, inp, out):
                # Binary code: 1 if activation > 0, else 0
                # Flatten spatial dims: (B, C, H, W) → (B, C*H*W)
                code = (out.detach() > 0).float()
                b = code.size(0)
                activation_codes.append(code.view(b, -1))
            return hook_fn

        for module in model.modules():
            if isinstance(module, (nn.ReLU, nn.ReLU6, nn.Hardswish)):
                hooks.append(module.register_forward_hook(make_hook()))

        with torch.no_grad():
            try:
                _ = model(x)
            except Exception:
                for h in hooks: h.remove()
                return 0.0

        for h in hooks: h.remove()

        if not activation_codes:
            return 0.0

        # Concatenate all activation codes: (B, total_neurons)
        try:
            codes = torch.cat(activation_codes, dim=1).cpu().numpy()
        except Exception:
            return 0.0

        if codes.shape[1] == 0:
            return 0.0

        # Kernel matrix K[i,j] = number of neurons that fire the same way
        # for inputs i and j.  K = codes @ codes.T (dot product of binary vecs)
        B = codes.shape[0]
        K = codes @ codes.T  # (B, B)

        # Log determinant of K — measures how many "dimensions" are active
        # A higher log det means the batch maps to a more diverse set of
        # activation patterns → higher expressivity.
        try:
            # Add small diagonal for numerical stability
            K += np.eye(B) * 1e-5
            sign, logdet = np.linalg.slogdet(K)
            if sign <= 0:
                return 0.0
            return float(logdet)
        except Exception:
            return 0.0


# ── Proxy 4: Jacobian condition number ───────────────────────────────────────

class JacobianProxy(ZeroCostProxy):
    """
    Jacobian-based trainability proxy (Mellor et al., extended).

    Computes the Jacobian of the backbone's output activations with
    respect to the input, then measures its condition number.

    A well-conditioned Jacobian (condition number near 1) means:
      - Gradients flow uniformly through the network
      - No vanishing or exploding gradient directions
      - More stable training → better final accuracy

    Score = -log(condition_number)   (negative because lower is better,
    we want score to be higher = better, so we negate)

    Alternatively: score = 1 / condition_number

    Requires: one forward+backward pass per input sample (slow for large
    batches — use small batch_size=4).
    """
    name = "jacobian"

    def __init__(self, n_samples: int = 4):
        self.n_samples = n_samples

    def score(self, model, x, targets=None):
        model.eval()
        x_small = x[:self.n_samples].requires_grad_(True)

        # Collect output of last backbone layer
        backbone_out = []
        hooks = []

        def hook_fn(module, inp, out):
            backbone_out.append(out)

        # Hook onto the last non-head layer
        model_layers = list(model.model.children())
        if len(model_layers) < 2:
            return 0.0

        h = model_layers[-2].register_forward_hook(hook_fn)
        hooks.append(h)

        try:
            _ = model(x_small)
        except Exception:
            for h in hooks: h.remove()
            return 0.0

        for h in hooks: h.remove()

        if not backbone_out:
            return 0.0

        out = backbone_out[-1]
        if isinstance(out, (list, tuple)):
            out = out[0]

        # Flatten output: (n_samples, -1)
        out_flat = out.view(self.n_samples, -1)
        n_out = out_flat.shape[1]

        if n_out == 0:
            return 0.0

        # Sample random projection directions to keep computation tractable
        n_proj = min(n_out, 32)
        proj_directions = torch.randn(n_out, n_proj, device=x.device)

        # Build approximate Jacobian by projecting onto random directions
        jacobians = []
        for i in range(n_proj):
            if x_small.grad is not None:
                x_small.grad.zero_()
            v = proj_directions[:, i]
            projected = (out_flat * v.unsqueeze(0)).sum()
            try:
                grad = torch.autograd.grad(
                    projected, x_small,
                    retain_graph=(i < n_proj - 1),
                    create_graph=False,
                    allow_unused=True,
                )[0]
                if grad is not None:
                    jacobians.append(grad.view(self.n_samples, -1))
            except Exception:
                continue

        if len(jacobians) == 0:
            return 0.0

        # J shape: (n_proj, n_samples * input_flat)
        J = torch.stack([j.flatten() for j in jacobians])

        try:
            # Condition number via SVD
            _, s, _ = torch.linalg.svd(J, full_matrices=False)
            s = s[s > 1e-10]
            if len(s) < 2:
                return 0.0
            cond = (s.max() / s.min()).item()
            # Lower condition number = better trainability
            # Return negative log so higher score = better
            return float(-np.log(max(cond, 1.0)))
        except Exception:
            return 0.0


# ── Proxy 5: GradNorm ────────────────────────────────────────────────────────

class GradNorm(ZeroCostProxy):
    """
    Gradient norm proxy.

    Simply measures the L2 norm of all gradients after a single
    backward pass.  Large gradient norms at initialisation indicate
    the network is sensitive to its parameters — information can flow.

    Very fast, very simple, surprisingly competitive with more
    sophisticated proxies on detection benchmarks.

    Requires: one forward+backward pass with a few images.
    """
    name = "gradnorm"

    def score(self, model, x, targets=None):
        model.train()
        model.zero_grad()

        try:
            # Use backbone features as proxy loss
            acts = []

            def hook(m, i, o):
                acts.append(o)

            # Hook last backbone layer
            layers = list(model.model.children())
            if not layers: return 0.0
            h = layers[-2].register_forward_hook(hook)

            _ = model(x)
            h.remove()

            if not acts: return 0.0
            out = acts[-1]
            if isinstance(out, (list, tuple)):
                out = out[0]
            loss = out.abs().mean()
            loss.backward()

        except Exception:
            model.zero_grad()
            return 0.0

        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2

        model.zero_grad()
        return float(np.sqrt(grad_norm))


# ============================================================
# Analytical efficiency metrics (no forward pass needed)
# ============================================================

def compute_flops_macs(arch: Architecture,
                       imgsz: int = 128,
                       nc: int = 1) -> tuple[int, int]:
    """
    Compute FLOPs and MACs analytically.
    MACs = multiplications.  FLOPs ≈ 2 × MACs (multiply + accumulate).

    For Conv(Cin, Cout, K, K, H, W):
        MACs = Cout × H_out × W_out × Cin × K × K / groups

    For DSConv(Cin, Cout):
        MACs = Cin × H × W × K² / Cin   (DW)
             + Cin × H × W × Cout        (PW)
             = H × W × (Cin × K² + Cin × Cout)
    """
    h = imgsz
    total_macs = 0

    def conv_macs(ci, co, k, h, g=1):
        h_out = h // 1  # stride=1 usually
        return co * h_out * h_out * (ci // g) * k * k

    def ds_macs(ci, co, k, h):
        return h*h * (ci*k*k + ci*co)

    def inception_macs(ci, co, h, split_idx=0, bn=0.5):
        spl   = INCEPTION_SPLITS[split_idx]
        ratio = co / sum(spl)
        sc    = [max(4, int(s*ratio)) for s in spl]
        sc[-1]= co - sum(sc[:-1])
        b1,b2,b3,b4 = sc
        m2 = max(4,int(b2*bn)); m3 = max(4,int(b3*bn))
        macs = 0
        macs += conv_macs(ci, b1, 1, h)                  # B1: 1×1
        macs += conv_macs(ci, m2, 1, h) + ds_macs(m2,b2,3,h)  # B2
        macs += conv_macs(ci, m3, 1, h) + ds_macs(m3,m3,3,h) + ds_macs(m3,b3,3,h)  # B3
        macs += conv_macs(ci, b4, 1, h)                  # B4 after pool (same h)
        return macs

    def c2f_macs(ci, co, n, h, e=0.5):
        c = max(1, int(co*e))
        return (conv_macs(ci,2*c,1,h)
                + n*(conv_macs(c,c,3,h)+conv_macs(c,c,3,h))
                + conv_macs((2+n)*c,co,1,h))

    def module_macs(name, ci, co, n, h, split_idx=0, n_aware=3, bn_arg=0.5):
        if name == "Conv":       return conv_macs(ci,co,3,h) * n
        if name == "DSConv":     return ds_macs(ci,co,3,h) * n
        if name == "DSResBlock": return (ds_macs(ci,ci,3,h)*2) * n
        if name == "C2f":        return c2f_macs(ci,co,n_aware,h)
        if name == "C3":         return c2f_macs(ci,co,n_aware,h)  # similar
        if name == "GhostConv":  return conv_macs(ci,co//2,3,h)+ds_macs(co//2,co//2,5,h)
        if name == "RepConv":    return conv_macs(ci,co,3,h) * n   # fused at deploy
        if name == "Focus":      return conv_macs(ci*4,co,3,h//2)
        if name == "DSInception":return inception_macs(ci,co,h,split_idx,bn_arg)
        return conv_macs(ci,co,3,h) * n

    c = 3
    sn, _, _ = arch.stem_op()
    total_macs += module_macs(sn, c, arch.stem_ch, 1, h)
    c = arch.stem_ch
    h //= 2

    for (bn,ba,_), bch, bn_r, bn_a, sidx, cur_h in [
        (arch.b2_op(), arch.b2_ch, arch.b2_n, arch.b2_n_aware, arch.b2_inc_split_idx, h),
        (arch.b3_op(), arch.b3_ch, arch.b3_n, arch.b3_n_aware, arch.b3_inc_split_idx, h//2),
        (arch.b4_op(), arch.b4_ch, arch.b4_n, arch.b4_n_aware, arch.b4_inc_split_idx, h//4),
        (arch.b5_op(), arch.b5_ch, arch.b5_n, arch.b5_n_aware, arch.b5_inc_split_idx, h//8),
    ]:
        en  = bn_a if bn in N_AWARE_MODULES else bn_r
        ib  = float(ba[0]) if (bn=="DSInception" and ba) else 0.5
        total_macs += module_macs(bn, c, bch, en, cur_h, sidx, bn_a, ib)
        c = bch

    flops = total_macs * 2
    return flops, total_macs


# ============================================================
# Composite zero-cost score
# ============================================================

@dataclass
class ProxyScores:
    synflow:   float = 0.0
    snip:      float = 0.0
    naswot:    float = 0.0
    jacobian:  float = 0.0
    gradnorm:  float = 0.0
    params:    int   = 0
    flops:     int   = 0
    macs:      int   = 0
    composite: float = 0.0

    def to_dict(self):
        return asdict(self)


def normalise_scores(records: list[dict], key: str) -> None:
    """Normalise a score field to [0, 1] across all records in-place."""
    vals = [r["scores"][key] for r in records if r["scores"][key] != 0]
    if not vals: return
    mn, mx = min(vals), max(vals)
    if mx == mn: return
    for r in records:
        v = r["scores"][key]
        r["scores"][f"{key}_norm"] = (v - mn) / (mx - mn) if v != 0 else 0.0


def compute_composite(scores: ProxyScores,
                      params_limit: int,
                      flops_limit:  int,
                      # Weights for each proxy in composite score
                      w_synflow:  float = 0.30,
                      w_snip:     float = 0.20,
                      w_naswot:   float = 0.25,
                      w_jacobian: float = 0.15,
                      w_gradnorm: float = 0.10,
                      ) -> float:
    """
    Weighted combination of normalised proxy scores.

    Weights rationale:
      SynFlow (0.30)  — most reliable for trainability, no data needed
      NASWOT  (0.25)  — measures expressivity, good correlation with accuracy
      SNIP    (0.20)  — connection sensitivity, good for detection
      Jacobian(0.15)  — condition number, catches pathological architectures
      GradNorm(0.10)  — simple but still informative

    Note: this function expects pre-normalised scores (_norm fields).
    Call normalise_scores() on the full population first.
    """
    # These are set externally after normalisation
    return (w_synflow  * getattr(scores, "synflow_norm",  scores.synflow)
          + w_snip     * getattr(scores, "snip_norm",     scores.snip)
          + w_naswot   * getattr(scores, "naswot_norm",   scores.naswot)
          + w_jacobian * getattr(scores, "jacobian_norm",  scores.jacobian)
          + w_gradnorm * getattr(scores, "gradnorm_norm",  scores.gradnorm))


# ============================================================
# Zero-shot evaluator
# ============================================================

PROXIES = [SynFlow(), SNIP(), NASWOT(), JacobianProxy(), GradNorm()]


def evaluate_zero_shot(
    arch:        Architecture,
    imgsz:       int = 128,
    nc:          int = 1,
    batch_size:  int = 8,
    device:      str = "cpu",
    skip_proxies: list[str] = None,
) -> ProxyScores:
    """
    Score one architecture using all zero-cost proxies.
    Returns ProxyScores with raw (un-normalised) values.
    """
    scores = ProxyScores()
    skip   = set(skip_proxies or [])

    # Build model
    model = build_model(arch, nc=nc, imgsz=imgsz)
    if model is None:
        return scores

    model = model.to(device)
    scores.params = arch.parameters

    # Analytical metrics — no forward pass
    scores.flops, scores.macs = compute_flops_macs(arch, imgsz, nc)

    # Proxy metrics
    x = get_dummy_batch(imgsz, batch_size, device)

    for proxy in PROXIES:
        if proxy.name in skip:
            continue
        t0  = time.time()
        val = proxy(model, x)
        setattr(scores, proxy.name, val)
        elapsed = time.time() - t0

    # Clean up
    del model
    if device != "cpu":
        torch.cuda.empty_cache()

    return scores


# ============================================================
# Main search
# ============================================================

def zero_shot_nas(
    data_yaml:       str   = "person.yaml",
    ram_limit:       int   = 500_000,
    flash_limit:     int   = 600_000,
    params_limit:    int   = 200_000,
    flops_limit:     int   = 500_000_000,
    n_samples:       int   = 500,
    imgsz:           int   = 128,
    nc:              int   = 1,
    device:          str   = "cpu",
    seed:            int   = 42,
    results_file:    str   = "zs_results.json",
    top_k:           int   = 10,
    verify_topk:     int   = 0,    # if >0, run proxy training on top-K
    proxy_epochs:    int   = 3,
    skip_proxies:    list  = None,
    proxy_weights:   dict  = None,
) -> list[Architecture]:
    """
    Zero-shot NAS: score architectures without training.

    1. Sample n_samples random architectures that pass hardware constraints.
    2. Score each with zero-cost proxies (seconds per architecture).
    3. Rank by composite score.
    4. Optionally verify top_k with proxy training (from nas.py).
    5. Return ranked list.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    pw = proxy_weights or {}
    skip = skip_proxies or []

    print(f"\n{'='*65}")
    print(f"Zero-Shot NAS")
    print(f"  Samples: {n_samples}  Top-K: {top_k}  Device: {device}")
    print(f"  Constraints: RAM≤{ram_limit//1024}KB  Params≤{params_limit//1000}K")
    print(f"  Proxies: {[p.name for p in PROXIES if p.name not in skip]}")
    print(f"{'='*65}\n")

    # ── Sample candidates ─────────────────────────────────────────────
    counter   = [0]
    candidates: list[Architecture] = []

    # Always include the known-good default
    default = Architecture()
    default.eval_id = "zs_default"
    if passes_constraints(default, ram_limit, flash_limit, imgsz, nc):
        candidates.append(default)

    pbar = tqdm(total=n_samples, desc="Sampling valid architectures")
    attempts = 0
    while len(candidates) < n_samples and attempts < n_samples * 20:
        attempts += 1
        a = Architecture.random()
        if passes_constraints(a, ram_limit, flash_limit, imgsz, nc):
            counter[0] += 1
            a.eval_id = f"zs_{counter[0]:04d}"
            candidates.append(a)
            pbar.update(1)
    pbar.close()

    print(f"Sampled {len(candidates)} valid architectures "
          f"({attempts} attempts)\n")

    # ── Score each candidate ──────────────────────────────────────────
    records = []
    pbar = tqdm(candidates, desc="Zero-shot scoring")

    for arch in pbar:
        pbar.set_postfix({"id": arch.eval_id})
        scores = evaluate_zero_shot(
            arch, imgsz=imgsz, nc=nc,
            device=device, skip_proxies=skip)

        # Also check params limit analytically
        if scores.params > params_limit:
            continue

        records.append({
            "arch":   arch.to_dict(),
            "scores": scores.to_dict(),
        })

    print(f"\nScored {len(records)} candidates")

    # ── Normalise all proxy scores across population ──────────────────
    for proxy in PROXIES:
        if proxy.name not in skip:
            normalise_scores(records, proxy.name)

    # ── Compute composite score ───────────────────────────────────────
    for r in records:
        s = r["scores"]
        # Build a temporary ProxyScores with normalised values
        ps = ProxyScores(**{k: v for k, v in s.items()
                            if k in ProxyScores.__dataclass_fields__})
        # Attach normalised versions
        for proxy in PROXIES:
            norm_key = f"{proxy.name}_norm"
            if norm_key in s:
                setattr(ps, norm_key, s[norm_key])

        composite = compute_composite(
            ps, params_limit, flops_limit,
            **{f"w_{k}": v for k, v in pw.items()})
        r["scores"]["composite"] = composite
        r["arch"]["proxy_map50"] = composite  # reuse field for compatibility

    # ── Rank by composite ─────────────────────────────────────────────
    records.sort(key=lambda r: r["scores"]["composite"], reverse=True)

    # ── Save results ──────────────────────────────────────────────────
    with open(results_file, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Results saved → {results_file}")

    # ── Print top-K ───────────────────────────────────────────────────
    _print_top_k(records, top_k, skip)

    # ── Optional: verify top-K with proxy training ────────────────────
    top_archs = [Architecture.from_dict(r["arch"]) for r in records[:top_k]]

    if verify_topk > 0 and data_yaml:
        print(f"\nVerifying top {verify_topk} with {proxy_epochs}-epoch proxy training...")

        for arch in top_archs[:verify_topk]:
            arch.proxy_map50 = evaluate_candidate(
                arch, data_yaml, proxy_epochs, imgsz,
                device, "zs_verify", nc)
            print(f"  {arch.eval_id}  "
                  f"composite={arch.parameters}  "
                  f"proxy_mAP={arch.proxy_map50:.4f}")

        top_archs.sort(key=lambda a: a.proxy_map50, reverse=True)
        print(f"\nBest after verification: {top_archs[0].eval_id}  "
              f"mAP={top_archs[0].proxy_map50:.4f}")

    return top_archs


def _print_top_k(records: list, k: int, skip: list) -> None:
    active = [p.name for p in PROXIES if p.name not in skip]
    header = (f"{'ID':<12} {'comp':>6} "
              + "  ".join(f"{p[:6]:>6}" for p in active)
              + f"  {'params':>7} {'MACs/M':>7} {'RAM':>5}")
    print(f"\nTop {k} architectures by composite score:")
    print(header); print("-" * len(header))

    for r in records[:k]:
        s = r["scores"]
        a = r["arch"]
        proxy_vals = "  ".join(
            f"{s.get(f'{p}_norm', s.get(p, 0)):>6.3f}" for p in active)
        print(f"{a['eval_id']:<12} "
              f"{s['composite']:>6.3f}  "
              f"{proxy_vals}  "
              f"{s['params']:>7,} "
              f"{s['macs']//1_000_000:>6}M "
              f"{a['est_ram']//1024:>4}K")


# ============================================================
# Correlation analysis (requires paired zero-shot + training results)
# ============================================================

def analyse_correlation(zs_file: str, training_file: str) -> None:
    """
    Given a zero-shot results file and a training NAS results file
    (from nas.py), compute the Spearman rank correlation between
    each proxy score and the actual proxy_map50 from training.

    This tells you which proxy is most predictive for your specific
    search space and dataset.
    """
    from scipy.stats import spearmanr

    with open(zs_file)       as f: zs_data  = json.load(f)
    with open(training_file) as f: tr_data  = json.load(f)

    # Match by eval_id
    tr_map = {r["eval_id"]: r["proxy_map50"] for r in tr_data
              if "proxy_map50" in r}

    proxy_names = [p.name for p in PROXIES] + ["composite"]
    results = {p: [] for p in proxy_names}
    actual_maps = []

    for r in zs_data:
        eid = r["arch"]["eval_id"]
        if eid not in tr_map:
            continue
        actual_maps.append(tr_map[eid])
        for p in proxy_names:
            results[p].append(r["scores"].get(p, 0))

    if len(actual_maps) < 5:
        print("Not enough matched architectures for correlation analysis.")
        return

    print(f"\nSpearman rank correlation (zero-shot proxy vs actual mAP@50)")
    print(f"N = {len(actual_maps)} matched architectures")
    print(f"{'Proxy':<12} {'Spearman ρ':>10} {'p-value':>10}")
    print("-" * 35)

    for p in proxy_names:
        if not results[p]: continue
        rho, pval = spearmanr(results[p], actual_maps)
        print(f"  {p:<12} {rho:>10.4f} {pval:>10.4f}"
              + (" ← best" if rho == max(
                  spearmanr(results[q], actual_maps)[0]
                  for q in proxy_names if results[q]) else ""))


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Zero-Shot NAS for TinyissimoYOLO")
    p.add_argument("--data",          default="person.yaml")
    p.add_argument("--ram_limit",     type=int,   default=300_000)
    p.add_argument("--flash_limit",   type=int,   default=500_000)
    p.add_argument("--params_limit",  type=int,   default=200_000)
    p.add_argument("--flops_limit",   type=int,   default=500_000_000)
    p.add_argument("--samples",       type=int,   default=500)
    p.add_argument("--imgsz",         type=int,   default=96)
    p.add_argument("--nc",            type=int,   default=1)
    p.add_argument("--device",        default="cpu",
                   help="cpu for zero-shot (no GPU memory needed), "
                        "or cuda:0 for faster Jacobian/SNIP")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--results",       default="zs_results.json")
    p.add_argument("--top_k",         type=int,   default=10)
    p.add_argument("--verify_topk",   type=int,   default=0,
                   help="If >0, verify top-K with N-epoch proxy training")
    p.add_argument("--proxy_epochs",  type=int,   default=3)
    p.add_argument("--skip_proxies",  nargs="+",  default=[],
                   help="Proxy names to skip, e.g. --skip jacobian")
    p.add_argument("--mode",
                   choices=["search", "analyse", "correlate"],
                   default="search")
    p.add_argument("--training_results", default=None,
                   help="nas.py results file for correlation analysis")

    args = p.parse_args()

    if args.mode == "search":
        zero_shot_nas(
            data_yaml    = args.data,
            ram_limit    = args.ram_limit,
            flash_limit  = args.flash_limit,
            params_limit = args.params_limit,
            flops_limit  = args.flops_limit,
            n_samples    = args.samples,
            imgsz        = args.imgsz,
            nc           = args.nc,
            device       = args.device,
            seed         = args.seed,
            results_file = args.results,
            top_k        = args.top_k,
            verify_topk  = args.verify_topk,
            proxy_epochs = args.proxy_epochs,
            skip_proxies = args.skip_proxies,
        )

    elif args.mode == "correlate":
        if not args.training_results:
            raise ValueError("--training_results required for correlate mode")
        analyse_correlation(args.results, args.training_results)