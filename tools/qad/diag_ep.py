"""QAD_DIAG=1 环境门控的诊断插桩：逐步给出「集合通信逐项耗时 + 层重算耗时 + 显存」分解。

契约：
- QAD_DIAG 未设时严格 no-op（集合通信包装器不安装；note/snapshot/reset/step_report 全为空操作）。
- 仅用于定位 backward 单调劣化（fwd 恒定 ~21s 而 bwd 33s→461s）；包装器在每次集合通信前后各做一次
  torch.npu.synchronize()，会引入同步开销 —— 诊断专用，勿用于正式训练。
"""
from __future__ import annotations

import os
import resource
import time

import torch

ENABLED = os.environ.get("QAD_DIAG") == "1"

_OPS = (
    "all_to_all_single",
    "all_gather_into_tensor",
    "_all_gather_base",
    "reduce_scatter_tensor",
    "_reduce_scatter_base",
    "all_reduce",
    "broadcast",
    "barrier",
)


class _Stats:
    def __init__(self):
        self.reset_step()

    def reset_step(self):
        self.ops = {n: {"t": 0.0, "calls": 0, "bytes": 0} for n in _OPS}
        self.a2a_in_max = None
        self.a2a_out_max = None
        self.a2a_patterns = set()
        self.layer_fwd_time = 0.0
        self.layer_fwd_calls = 0


STATS = _Stats() if ENABLED else None
_INSTALLED = False


def _sync():
    try:
        torch.npu.synchronize()
    except Exception:
        pass


def _tbytes(t):
    try:
        return int(t.numel()) * int(t.element_size())
    except Exception:
        return 0


def _acc(name, dt, nbytes=0):
    if STATS is None:
        return
    d = STATS.ops.get(name)
    if d is None:
        d = STATS.ops[name] = {"t": 0.0, "calls": 0, "bytes": 0}
    d["t"] += dt
    d["calls"] += 1
    d["bytes"] += nbytes


def _note_a2a(args, kwargs):
    if STATS is None:
        return
    outs = kwargs.get("output_split_sizes")
    ins = kwargs.get("input_split_sizes")
    if outs is None and len(args) > 2:
        outs = args[2]
    if ins is None and len(args) > 3:
        ins = args[3]
    ot = tuple(outs) if outs is not None else None
    it = tuple(ins) if ins is not None else None
    if it:
        m = max(it)
        STATS.a2a_in_max = m if STATS.a2a_in_max is None else max(STATS.a2a_in_max, m)
    if ot:
        m = max(ot)
        STATS.a2a_out_max = m if STATS.a2a_out_max is None else max(STATS.a2a_out_max, m)
    STATS.a2a_patterns.add((it, ot))


def _wrap(name, orig, in_idx):
    def _w(*args, **kwargs):
        _sync()
        t0 = time.time()
        in_t = args[in_idx] if (in_idx is not None and in_idx < len(args)) else None
        nb = _tbytes(in_t)
        if name == "all_to_all_single":
            _note_a2a(args, kwargs)
        r = orig(*args, **kwargs)
        _sync()
        _acc(name, time.time() - t0, nb)
        return r

    return _w


def install(log_fn=print):
    global _INSTALLED
    if not ENABLED or _INSTALLED:
        return
    import torch.distributed as dist

    _INSTALLED = True
    specs = [
        ("all_to_all_single", 1),
        ("all_gather_into_tensor", 1),
        ("_all_gather_base", 1),
        ("reduce_scatter_tensor", 1),
        ("_reduce_scatter_base", 1),
        ("all_reduce", 0),
        ("broadcast", 0),
        ("barrier", None),
    ]
    n = 0
    for nm, idx in specs:
        fn = getattr(dist, nm, None)
        if fn is None:
            continue
        setattr(dist, nm, _wrap(nm, fn, idx))
        n += 1
    log_fn(f"[diag] installed {n} collective wrappers (QAD_DIAG=1)")


def note_layer_fwd(dt_s):
    if STATS is None:
        return
    STATS.layer_fwd_time += dt_s
    STATS.layer_fwd_calls += 1


def snapshot_layer_fwd():
    if STATS is None:
        return 0.0
    return STATS.layer_fwd_time


def reset():
    if STATS is None:
        return
    STATS.reset_step()


def _host_avail_gib():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576
    except Exception:
        pass
    return -1.0


def _peak_rss_gib():
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    except Exception:
        return -1.0


def _mem():
    try:
        a = torch.npu.memory_allocated() / (2 ** 30)
        r = torch.npu.memory_reserved() / (2 ** 30)
        st = torch.npu.memory_stats() or {}
        return a, r, int(st.get("num_alloc_retries", 0)), int(st.get("num_ooms", 0))
    except Exception:
        return -1.0, -1.0, 0, 0


def step_report(step, phases):
    if STATS is None:
        return ""
    o = STATS.ops
    zero = {"t": 0.0, "calls": 0, "bytes": 0}
    a2a = o.get("all_to_all_single", zero)
    ag = o.get("all_gather_into_tensor", zero)["t"] + o.get("_all_gather_base", zero)["t"]
    rs = (o.get("reduce_scatter_tensor", zero)["t"]
          + o.get("_reduce_scatter_base", zero)["t"])
    ar = o.get("all_reduce", zero)["t"]
    bc = o.get("broadcast", zero)["t"]
    bar = o.get("barrier", zero)["t"]
    coll = a2a["t"] + ag + rs + ar + bc + bar

    total_lf = STATS.layer_fwd_time
    fwd_lf = float(phases.get("layer_fwd_at_fwd_end", 0.0) or 0.0)
    recomp = max(total_lf - fwd_lf, 0.0)
    a, r, retries, ooms = _mem()
    in_max = "-" if STATS.a2a_in_max is None else str(STATS.a2a_in_max)
    out_max = "-" if STATS.a2a_out_max is None else str(STATS.a2a_out_max)
    return (
        f"[diag step={step}] fwd={phases.get('fwd', 0):.1f}s bwd={phases.get('bwd', 0):.1f}s "
        f"step={phases.get('step', 0):.1f}s | "
        f"a2a={a2a['t']:.1f}s(calls={a2a['calls']},in_max={in_max},out_max={out_max},"
        f"bytes={a2a['bytes'] / 2**30:.2f}GiB,pat={len(STATS.a2a_patterns)}) "
        f"ag={ag:.1f}s rs={rs:.1f}s ar={ar:.1f}s bc={bc:.1f}s bar={bar:.1f}s "
        f"collsum={coll:.1f}s | "
        f"layer_fwd_total={total_lf:.1f}s(calls={STATS.layer_fwd_calls},"
        f"fwd_phase={fwd_lf:.1f}s,recompute~{recomp:.1f}s) | "
        f"mem alloc/reserved={a:.1f}/{r:.1f}GiB retries={retries} ooms={ooms} "
        f"host_avail={_host_avail_gib():.0f}GiB host_rss={_peak_rss_gib():.1f}GiB"
    )
