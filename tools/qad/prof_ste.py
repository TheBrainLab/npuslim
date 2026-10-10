"""QAD STE profiling instrumentation (env: QAD_PROFILE=1).

Monkeypatches npuslim.qad.ste_experts to count/time:
  - STEQwen3Experts._effective_all  (call count + wall time + H2D bytes)
  - STEQwen3Experts.forward          (per-layer wall time)
  - _gather_full_from_shard          (gather call count + wall time + all_gather bytes)

Per-step phase timing (fetch/forward/backward/step) is handled in ds_train_glm.py
(env-gated). This module only owns the STE-layer instrumentation + the shared
STATS accumulator that the trainer flushes at step boundaries.

Importing this module installs the patches as a side effect. No-op if
QAD_PROFILE != "1".
"""
from __future__ import annotations

import os
import time

import torch.distributed as dist

from npuslim.qad import ste_experts as _se

ENABLED = os.environ.get("QAD_PROFILE") == "1"


class _Stats:
    def __init__(self):
        self.reset_step()

    def reset_step(self):
        self.eff_all_calls = 0
        self.eff_all_time = 0.0
        self.gather_calls = 0
        self.gather_time = 0.0
        self.bytes_allgather = 0          # bytes shipped over all_gather (per-rank recv × world)
        self.bytes_h2d = 0                # bytes moved CPU→NPU after gather
        self.layer_fwd_calls = 0
        self.layer_fwd_time = 0.0
        self.layer_log: list[tuple] = []  # (qad_idx, dt_s, eff_calls, gather_time_s)


STATS = _Stats()


def install():
    if not ENABLED:
        return

    _orig_gather = _se._gather_full_from_shard

    def _prof_gather(shard, ds_numel, ds_shape):
        t0 = time.time()
        out = _orig_gather(shard, ds_numel, ds_shape)
        dt = time.time() - t0
        STATS.gather_calls += 1
        STATS.gather_time += dt
        try:
            world = dist.get_world_size()
        except Exception:
            world = 1
        STATS.bytes_allgather += shard.numel() * shard.element_size() * world
        return out

    _se._gather_full_from_shard = _prof_gather

    _orig_eff_all = _se.STEQwen3Experts._effective_all

    def _prof_eff_all(self, proj):
        STATS.eff_all_calls += 1
        t0 = time.time()
        out = _orig_eff_all(self, proj)
        STATS.eff_all_time += time.time() - t0
        # H2D bytes: only when zero3_gather path ran (gloo gather → CPU → .to(npu))
        if self.zero3_gather:
            w = self.experts.gate_up_proj if proj == "gate_up" else self.experts.down_proj
            if getattr(w, "ds_tensor", None) is not None:
                STATS.bytes_h2d += int(getattr(w, "ds_numel", w.numel())) * 2  # bf16
        return out

    _se.STEQwen3Experts._effective_all = _prof_eff_all

    _orig_fwd = _se.STEQwen3Experts.forward

    def _prof_fwd(self, hidden_states, top_k_index, top_k_weights):
        ec0 = STATS.eff_all_calls
        gt0 = STATS.gather_time
        t0 = time.time()
        out = _orig_fwd(self, hidden_states, top_k_index, top_k_weights)
        dt = time.time() - t0
        STATS.layer_fwd_calls += 1
        STATS.layer_fwd_time += dt
        qidx = getattr(self, "_qad_idx", -1)
        STATS.layer_log.append((qidx, dt, STATS.eff_all_calls - ec0,
                                STATS.gather_time - gt0))
        return out

    _se.STEQwen3Experts.forward = _prof_fwd


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PiB"


def flush_step(rank: int, step: int, phase_times: dict | None = None) -> str:
    """Return a formatted summary line; reset per-step counters."""
    s = STATS
    line = (f"[prof step={step} r{rank}] "
            f"eff_all_calls={s.eff_all_calls} eff_all_time={s.eff_all_time:.1f}s "
            f"gather_calls={s.gather_calls} gather_time={s.gather_time:.1f}s "
            f"allgather={fmt_bytes(s.bytes_allgather)} h2d={fmt_bytes(s.bytes_h2d)} "
            f"layer_fwd_calls={s.layer_fwd_calls} layer_fwd_time={s.layer_fwd_time:.1f}s")
    if phase_times:
        line += " phases=" + " ".join(f"{k}={v:.1f}s" for k, v in phase_times.items())
    # per-layer detail: first 5 + last 3 (the streaming heart of the profile)
    ll = s.layer_log
    if ll:
        head = ", ".join(f"L{q}={dt:.2f}s(ea={ea},gt={gt:.2f}s)"
                         for q, dt, ea, gt in ll[:5])
        tail = ", ".join(f"L{q}={dt:.2f}s(ea={ea},gt={gt:.2f}s)"
                         for q, dt, ea, gt in ll[-3:])
        line += f"\n  layers[0:5]: {head}"
        if len(ll) > 8:
            line += f"\n  layers[-3:]: {tail}"
    s.reset_step()
    return line


install()
