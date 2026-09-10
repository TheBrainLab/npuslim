"""M4 teacher 缓存解析器单测——合成 shard 验证 schema/转换/读取计划/KL 端到端。

合成缓存口径：K=5（缓存真实 K=100，逻辑相同）、2 shard、3 样本（跨 shard），
覆盖：actual 在/不在 top-K、id 重复、shard 切换、LRU 淘汰、行对齐自校验、
plan_epoch 聚簇、schema 断言、fetch → kl_distill_loss 前向+backward。
运行：/data/yult/miniconda3/envs/npuslim/bin/python tests/qad/test_teacher_cache.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from npuslim.qad.kl_loss import kl_distill_loss
from npuslim.qad.teacher_cache import TeacherCache, merge_actual_topk

K = 5
VOCAB = 512
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


# --------------------------------------------------------------------------- #
# merge_actual_topk 语义
# --------------------------------------------------------------------------- #
def test_merge() -> None:
    # case1: actual 已在 top-K（重复出现，去重保集合）
    ids = np.array([[7, 3, 1, 7, 2, 9]], np.int32)     # col0=7, col1=3..col5=9 含 7
    lps = np.array([[-0.1, -0.5, -1.0, -0.1, -2.0, -3.0]], np.float32)
    t_id, t_lp = merge_actual_topk(ids, lps)
    check("merge.in_topk", t_id.tolist() == [[3, 1, 7, 2, 9]]
          and np.allclose(t_lp, [[-0.5, -1.0, -0.1, -2.0, -3.0]]))
    # 注意 col1=7(lp -0.1) 与 col0 重复——合并集合应为 {3,1,7,2,9}，此处 cand 顺序保持

    # case2: actual 不在 top-K 且 lp0 最大 → 挤掉第 K 名
    ids = np.array([[42, 3, 1, 5, 2, 9]], np.int32)
    lps = np.array([[-0.05, -0.5, -1.0, -1.5, -2.0, -3.0]], np.float32)
    t_id, _ = merge_actual_topk(ids, lps)
    check("merge.evict_last", 42 in t_id[0].tolist() and 9 not in t_id[0].tolist())

    # case3: actual 不在且 lp0 小于 min → 落选，top-K 原样
    ids = np.array([[42, 3, 1, 5, 2, 9]], np.int32)
    lps = np.array([[-9.9, -0.5, -1.0, -1.5, -2.0, -3.0]], np.float32)
    t_id, t_lp = merge_actual_topk(ids, lps)
    check("merge.dropped", t_id.tolist() == [[3, 1, 5, 2, 9]]
          and np.allclose(t_lp, [[-0.5, -1.0, -1.5, -2.0, -3.0]]))

    # case4: 无序候选列（min 取真 min，不假设降序）
    ids = np.array([[42, 9, 1, 5, 2, 3]], np.int32)
    lps = np.array([[-0.7, -3.0, -1.0, -1.5, -2.0, -0.5]], np.float32)
    t_id, _ = merge_actual_topk(ids, lps)
    check("merge.unsorted_min", 42 in t_id[0].tolist() and 9 not in t_id[0].tolist())

    # 批量正确性: 向量化 vs 逐行参考
    rng = np.random.default_rng(7)
    L = 64
    ids = np.empty((L, K + 1), np.int32)
    lps = rng.normal(-2, 2, (L, K + 1)).astype(np.float32)
    for i in range(L):
        cand = rng.choice(VOCAB, K, replace=False)
        ids[i] = np.concatenate([[cand[0] if rng.random() < 0.5 else VOCAB + 7], cand])
        if ids[i, 0] == ids[i, 1]:                       # 造 actual 在 topk 的情况
            lps[i, 0] = lps[i, 1]
    t_id, t_lp = merge_actual_topk(ids, lps)
    ok = True
    for i in range(L):
        pool = {}
        for tid, lp in zip(ids[i], lps[i]):
            pool[int(tid)] = max(lp, pool.get(int(tid), -np.inf))   # 同 token 同 lp
        top = sorted(pool.items(), key=lambda kv: -kv[1])[:K]
        ref_set = {t for t, _ in top}
        ok &= set(t_id[i].tolist()) == ref_set
    check("merge.batch_vs_ref", ok, f"L={L}")


# --------------------------------------------------------------------------- #
# 合成缓存 fixture
# --------------------------------------------------------------------------- #
def build_cache(tmp: Path) -> tuple[TeacherCache, dict]:
    """2 shard / 3 样本 / K=5。样本 token 序列与缓存行严格按写入端语义构造。"""
    rng = np.random.default_rng(42)
    samples = {}                                        # id -> (token 序列 [n], row 数据)
    index_rows, shard_data = [], {}                     # shard -> list[(ids[K+1], lps[K+1])]

    def make_sample(sid, n_tokens, shard, force_actural_out=False):
        """构造 n-1 行缓存数据；每行: col0=actual(token[i+1]), col1..K=top-K 候选。"""
        toks = rng.integers(0, VOCAB, n_tokens).astype(np.int32)
        rows = []
        for i in range(n_tokens - 1):
            cand = rng.choice(VOCAB, K, replace=False)
            lp = np.linspace(-0.5, -4.0, K).astype(np.float32)   # 确定性降序
            if force_actural_out and i == 0:            # actual 不在 top-K、lp 高于全部候选
                cand = np.where(cand == toks[i + 1], (cand + 1) % VOCAB, cand)
                act_lp = 0.0
            else:
                act_lp = float(rng.normal(-3, 0.5))
            rows.append((np.concatenate([[toks[i + 1]], cand]),
                         np.concatenate([[act_lp], lp]).astype(np.float32)))
        shard_data.setdefault(shard, []).extend(rows)
        index_rows.append({"id": sid, "shard": shard, "row": -1, "n_tokens": n_tokens,
                           "logit_rows": n_tokens - 1, "top_k": K,
                           "file": "f.jsonl", "scenario": "s"})
        samples[sid] = toks

    make_sample("s1", 8, "shard_00000")
    make_sample("s2", 5, "shard_00000", force_actural_out=True)
    make_sample("s3", 6, "shard_00001")

    # 按 shard 落盘并回填 row
    for shard, rows in shard_data.items():
        ids = np.stack([r[0] for r in rows]).astype(np.int32)
        lps = np.stack([r[1] for r in rows]).astype(np.float32)
        np.savez(tmp / f"{shard}.npz", token_ids=ids, logits=lps)
    offset = {}
    for r in index_rows:
        r["row"] = offset.get(r["shard"], 0)
        offset[r["shard"]] = offset.get(r["shard"], 0) + r["logit_rows"]
    with open(tmp / "index.jsonl", "w") as f:
        for r in index_rows:
            f.write(json.dumps(r) + "\n")
    (tmp / "teacher_cfg.json").write_text(json.dumps(
        {"model": "Qwen3-30B-A3B-Instruct-2507", "top_k": K, "temperature": 1.0,
         "logits_semantics": "log_softmax; col0=actual"}))
    return TeacherCache(tmp), samples


def test_reader() -> None:
    with tempfile.TemporaryDirectory() as td:
        cache, toks = build_cache(Path(td))

        check("reader.index_size", len(cache) == 3 and "s2" in cache and "s9" not in cache)

        # fetch: 行数、行对齐（actual_ids == token 序列[1:]）、转换口径
        f1 = cache.fetch("s1")
        check("fetch.shape", f1.topk_ids.shape == (7, K) and f1.actual_ids.shape == (7,))
        check("fetch.row_align", np.array_equal(f1.actual_ids, toks["s1"][1:]))

        f2 = cache.fetch("s2")      # 行 0 actual 挤掉 lp 最小候选（确定性发生）
        check("fetch.evicted", f2.topk_ids.shape == (4, K)
              and toks["s2"][1] in f2.topk_ids[0].tolist())

        # plan_epoch: 跨 shard 聚簇
        plan = cache.plan_epoch(["s3", "s1", "s2"])
        check("plan.cluster", [r.id for r in plan] == ["s1", "s2", "s3"])

        # LRU: max=1 时访问 s3 应淘汰 shard_00000
        cache.max = None
        c1 = TeacherCache(Path(td), max_cached_shards=1)
        c1.fetch("s1"); c1.fetch("s3")
        check("lru.evict", list(c1._shards) == ["shard_00001"])

        # 缺失 id 报错
        try:
            cache.plan_epoch(["nope"])
            check("plan.missing_key", False)
        except KeyError:
            check("plan.missing_key", True)

        # schema 断言: top_k 不符
        bad = Path(td) / "bad"
        bad.mkdir()
        (bad / "teacher_cfg.json").write_text(json.dumps({"top_k": 4}))
        (bad / "index.jsonl").write_text(
            json.dumps({"id": "x", "shard": "s", "row": 0, "n_tokens": 2,
                        "logit_rows": 1, "top_k": 5}) + "\n")
        try:
            TeacherCache(bad)
            check("schema.topk_mismatch", False)
        except ValueError:
            check("schema.topk_mismatch", True)


def test_e2e_kl() -> None:
    """fetch → kl_distill_loss 前向 + 梯度流（合成 logits，单样本 7 行）。"""
    with tempfile.TemporaryDirectory() as td:
        cache, _ = build_cache(Path(td))
        s = cache.fetch("s1")
        logits = torch.randn(7, VOCAB, requires_grad=True)
        loss = kl_distill_loss(logits, torch.from_numpy(s.topk_ids),
                               torch.from_numpy(s.topk_lps), VOCAB, tau=1.0)
        loss.backward()
        check("e2e.finite", bool(torch.isfinite(loss))
              and bool(torch.isfinite(logits.grad).all()),
              f"loss={loss.item():+.4f}")


if __name__ == "__main__":
    test_merge()
    test_reader()
    test_e2e_kl()
    print("-" * 76)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print("TEACHER CACHE PARSER PASSED")
