#!/usr/bin/env python3
"""analyze_v2.py — 容错解析（渐进尾部裁剪）+ 聚合分析。
用法: python3 analyze_v2.py <trace.json.rep> [topN]
"""
import json, sys, collections

path = sys.argv[1]
topn = int(sys.argv[2]) if len(sys.argv) > 2 else 45

blob = open(path, "rb").read()
sys.stderr.write(f"loaded {len(blob)}B\n")
data = None
for k in range(0, 64):
    try:
        data = json.loads(blob[: len(blob) - k] if k else blob)
        sys.stderr.write(f"json OK with trim={k}\n")
        break
    except Exception as exc:
        err = str(exc).split("\n")[0][:80]
        if k % 8 == 0 or k < 4:
            sys.stderr.write(f"trim={k}: {err}\n")
if data is None:
    sys.exit("parse failed up to trim=63")

ev = data if isinstance(data, list) else data.get("traceEvents", [])
sys.stderr.write(f"events={len(ev)}\n")
agg = collections.defaultdict(lambda: [0.0, 0])
tmin = tmax = None
for e in ev:
    if e.get("ph") != "X":
        continue
    cat = e.get("cat", "?")
    name = e.get("name", "?")
    try:
        dur = float(e.get("dur", 0) or 0)
        ts = float(e.get("ts", 0) or 0)
    except (TypeError, ValueError):
        continue
    a = agg[(cat, name)]
    a[0] += dur
    a[1] += 1
    if tmin is None or ts < tmin:
        tmin = ts
    e_end = ts + dur
    if tmax is None or e_end > tmax:
        tmax = e_end

wall = (tmax - tmin) / 1e6 if tmin is not None else 0.0
cat_tot = collections.defaultdict(float)
for (cat, name), (tot, cnt) in agg.items():
    cat_tot[cat] += tot
print(f"== span={wall:.1f}s events={len(ev)} ==")
for cat, tot in sorted(cat_tot.items(), key=lambda kv: -kv[1]):
    print(f"  cat[{cat}] total={tot/1e6:.1f}s")
print()

def dump(rows, title, n):
    print(f"== {title} ==")
    for (cat, name), (tot, cnt) in rows[:n]:
        avg = tot / cnt / 1e3
        print(f"{tot/1e6:12.2f}s x{cnt:<8d} {avg:10.2f}ms/avg [{cat}] {name[:96]}")
    print()

rows_all = sorted(agg.items(), key=lambda kv: -kv[1][0])
dump(rows_all, f"ALL top{topn}", topn)
for cat in ("kernel", "hccl", "cpu_op", "Memcpy", "memcpy", "gpu_memcpy", "gpu_memset"):
    rows = [(k, v) for k, v in agg.items() if k[0] == cat]
    if rows:
        rows.sort(key=lambda kv: -kv[1][0])
        dump(rows, f"{cat} top15", 15)
