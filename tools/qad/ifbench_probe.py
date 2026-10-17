#!/usr/bin/env python3
"""IFBench probe: instruction-following with the official programmatic verifiers.

Protocol (fixed): chat endpoint, enable_thinking=false, temperature=0, max_tokens=512.
Metrics (official IFBench conventions):
  * prompt-level accuracy (loose)     : ALL constraints of the prompt satisfied
  * instruction-level accuracy (strict): fraction of individual constraints satisfied

Verifier code: allenai/IFBench (ifbench/{instructions,instructions_registry,
classic_instructions,instructions_util}.py) — loaded from the local package dir.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
IFBENCH_PKG = "/home/yult/llm_infer_workspace/ylt/glm52-qad/p2bcd_data/ifbench_pkg"


def post(url: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="/home/yult/llm_infer_workspace/ylt/glm52-qad/p2bcd_data/ifbench_test.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    sys.path.insert(0, IFBENCH_PKG)
    from ifbench import instructions_registry as reg  # noqa: E402
    import pyarrow.parquet as pq  # noqa: E402

    t = pq.read_table(args.data)
    rows = [{c: t.column(c)[i].as_py() for c in t.column_names} for i in range(t.num_rows)]
    if args.limit:
        rows = rows[:args.limit]

    def one(row):
        rec = {"key": row.get("key"), "ids": row.get("instruction_id_list")}
        try:
            resp = post(args.api, {"model": args.model, "temperature": 0, "max_tokens": 512,
                                   "chat_template_kwargs": {"enable_thinking": False},
                                   "messages": [{"role": "user", "content": row["prompt"]}]})
            content = resp["choices"][0]["message"]["content"] or ""
            per = []
            for iid, kw in zip(row["instruction_id_list"], row["kwargs"]):
                kwc = {k: v for k, v in (kw or {}).items() if v is not None}
                try:
                    instr = reg.INSTRUCTION_DICT[iid](instruction_id=iid)
                    instr.build_description(**kwc)
                    per.append(int(bool(instr.check_following(content))))
                except Exception:
                    per.append(0)
            rec.update({"per": per, "loose": int(all(per)) if per else 0,
                        "content": content[:300]})
        except Exception as e:
            rec.update({"error": repr(e)[:200], "per": [], "loose": 0})
        return rec

    from concurrent.futures import ThreadPoolExecutor, as_completed
    n = loose = 0
    inst_tot = inst_ok = 0
    err = 0
    with open(args.out, "w") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, r) for r in rows]
        for fut in as_completed(futs):
            rec = fut.result()
            n += 1
            if rec.get("error"):
                err += 1
            loose += rec.get("loose", 0)
            inst_tot += len(rec.get("per", []))
            inst_ok += sum(rec.get("per", []))
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if n % 50 == 0:
                print(f"  [{n}/{len(rows)}]", flush=True)

    print("=== summary ===")
    print(f"  prompt-level (loose, all constraints) = {loose}/{n} = {loose/max(n,1):.4f}")
    print(f"  instruction-level (strict) = {inst_ok}/{inst_tot} = {inst_ok/max(inst_tot,1):.4f} | err={err}")
    with open(args.out.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump({"n": n, "loose": loose, "inst_ok": inst_ok, "inst_tot": inst_tot, "err": err}, f, indent=1)


if __name__ == "__main__":
    main()
