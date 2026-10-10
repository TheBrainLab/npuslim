#!/usr/bin/env python3
"""Consolidate phase2b run metrics: diag lines (per-step) + mon samples (20s) -> CSV.

Usage: python3 phase2b_metrics.py [log_dir]   (default /home/yult/llm_infer_workspace/ylt)
Output: <log_dir>/phase2b_metrics.csv  (full rewrite, idempotent)
"""
# ===== Phase D 路径参数化（env 带默认回退；不改数值语义）=====
import os as _pd_os
SHARED = _pd_os.environ.get("SHARED", "/llm_infer_workspace")
SHARED_HOST = _pd_os.environ.get("SHARED_HOST", "/home/yult/llm_infer_workspace")
YLT = _pd_os.environ.get("YLT", SHARED + "/ylt")
NPUSLIM_SRC = _pd_os.environ.get("NPUSLIM_SRC", YLT + "/npuslim/src")
QAD_ROOT = _pd_os.environ.get("QAD_ROOT", YLT + "/glm52-qad")
# =========================================================
import csv
import glob
import re
import statistics
import sys

LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else f"{SHARED_HOST}/ylt"
TRAIN_LOG = f"{LOG_DIR}/phase2b_r0.log"
MON_LOG = f"{LOG_DIR}/phase2b_mon.log"
OUT = f"{LOG_DIR}/phase2b_metrics.csv"

diag_re = re.compile(
    r"^\[diag step=(\d+)\] fwd=([\d.]+)s bwd=([\d.]+)s step=([\d.]+)s \| "
    r"a2a=([\d.]+)s\(calls=(\d+),in_max=(\d+),out_max=(\d+),bytes=([\d.]+)GiB,pat=(\d+)\) "
    r".*?mem alloc/reserved=([\d.]+)/([\d.]+)GiB retries=(\d+) ooms=(\d+) "
    r"host_avail=(\d+)GiB host_rss=([\d.]+)GiB")

rows = {}
with open(TRAIN_LOG, errors="ignore") as f:
    for line in f:
        m = diag_re.match(line)
        if m:
            rows[int(m.group(1))] = m.groups()

mon = {}
with open(MON_LOG, errors="ignore") as f:
    next(f, None)
    for line in f:
        p = line.rstrip("\n").split("|")
        if len(p) < 7 or not p[1].isdigit():
            continue
        step = int(p[1])
        try:
            aic = [float(x) for x in p[3].split(",") if x.strip().isdigit()]
            pwr = [float(x) for x in p[4].split(",") if re.match(r"^[\d.]+$", x.strip())]
        except ValueError:
            continue
        agg = mon.setdefault(step, {"aic": [], "pwr": []})
        if aic:
            agg["aic"].append(statistics.mean(aic))
        if pwr:
            agg["pwr"].append(statistics.mean(pwr))

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step", "fwd_s", "bwd_s", "step_s", "a2a_s", "a2a_calls", "in_max",
                "out_max", "a2a_bytes_gib", "pat", "alloc_gib", "reserved_gib",
                "retries", "ooms", "host_avail_gib", "host_rss_gib",
                "aicore_mean_pct", "power_mean_w"])
    for step in sorted(rows):
        g = rows[step]
        a = mon.get(step, {})
        aic = f"{statistics.mean(a['aic']):.2f}" if a.get("aic") else ""
        pwr = f"{statistics.mean(a['pwr']):.1f}" if a.get("pwr") else ""
        w.writerow([step, *g[1:16], aic, pwr])

print(f"wrote {OUT}: {len(rows)} diag steps, {len(mon)} mon-sampled steps")
