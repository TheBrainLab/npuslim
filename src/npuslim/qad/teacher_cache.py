"""M4 teacher 缓存解析器——845GB 离线 top-K logit 缓存 → kl_distill_loss 输入。

缓存写入端：glm52-qad 9.2 `precompute_logits.py`（本地镜像 workspace/glm52-qad/
precompute/，Qwen3-30B §12.9 复用即此），schema（写入端代码 + teacher_cfg.json
权威声明）：

- shard npz：`token_ids [L, K+1] int32` + `logits [L, K+1] float32`（np.savez
  非压缩，单 shard ≈ 2M token 上限 → ~1.6GB）
- col 0 = actual token 的 logprob（无条件首列），col 1..K = top-K 候选
- logit 行 i（0-based）对应样本 token 位置 i+1 的上下文；BOS 位置 0 无行，
  logit_rows = n_tokens − 1
- index.jsonl 每样本：{id, shard, row, n_tokens, logit_rows, top_k, file, scenario}
- teacher_cfg.json：模型/top_k/temperature/logits_semantics（vocab 从模型侧
  config 读取，本模块不硬编码）

列 0 消费方式（对齐 InstinctRazor opd_score.py 先例）：actual 参与排序竞争、
不强制保留、不单独监督——转换层做 (actual ∪ top-K) 按 lp 降序取前 K 的合并。
KL 的 rest 项不受此影响（rest = top-K 集合外的质量，集合选择改变成员但质量
口径一致）。

IO 策略：npz 为 zip 容器不支持行级 mmap，按 shard 粒度惰性加载（~1.6GB/个），
LRU 缓存最近 shard；训练顺序必须按 `plan_epoch()` 聚簇（shard 内按 row 排序），
随机乱序访问会导致 shard 反复重载。

M4 首跑选型（2026-09-02 裁定）：Qwen3-30B 自蒸馏（缓存 §12.9 已验、W4A8 基线
损失画像现成、tokenizer 自洽）；GLM-5.2 等其 W4A8 基线完成后跟进。
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SampleRef:
    """index.jsonl 单条记录（样本 → 缓存物理位置）。"""

    id: str
    shard: str          # "shard_00000"
    row: int            # shard 内起始行
    n_tokens: int       # 样本 token 总数（含 BOS）
    logit_rows: int     # = n_tokens - 1
    top_k: int
    file: str
    scenario: str


@dataclass(frozen=True)
class FetchedSample:
    """单个样本的 teacher 数据（已转换为 KL 输入口径）。

    actual_ids = 样本 token 位置 1..n_tokens-1（col 0 还原），供 student 侧
    token 流对齐自校验（缓存语料与训练语料同源时两者应逐位一致）。
    topk_ids/topk_lps 行 i 与 student logits 行 i 对齐（均预测 token i+1）。
    """

    sample_id: str
    actual_ids: np.ndarray      # [logit_rows] int32
    topk_ids: np.ndarray        # [logit_rows, K] int64（kl_distill_loss 需 long）
    topk_lps: np.ndarray        # [logit_rows, K] float32


def merge_actual_topk(ids: np.ndarray, lps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[L, K+1] → ([L, K] int64, [L, K] float32)，对齐 opd_score 语义。

    数学等价于把 actual 与 top-K 候选合并后按 lp 降序取前 K：
    - actual 已在 top-K（实测 89.6% 行）：top-K 集合不变（同 token 同 lp）
    - actual 不在且 lp0 > min(top-K lp)：actual 挤掉 lp 最小的候选（argmin 槽位，
      候选列无序时 "替换末列" 不等价）
    - actual 不在且 lp0 <= min：落选（其质量归入 rest，口径一致）
    KL 只消费集合质量、不依赖顺序，故 "替换 argmin 槽位" 与 "全排序" 等价且免排序。
    """
    act_id, act_lp = ids[:, :1].astype(np.int64), lps[:, :1]
    cand_id, cand_lp = ids[:, 1:], lps[:, 1:]
    in_topk = (cand_id == act_id).any(axis=1, keepdims=True)
    amin = cand_lp.argmin(axis=1)                       # [L]，每行最小 lp 的槽位
    min_lp = np.take_along_axis(cand_lp, amin[:, None], axis=1)   # [L,1]
    replace = (~in_topk) & (act_lp > min_lp)
    out_id = cand_id.astype(np.int64, copy=True)
    out_lp = cand_lp.astype(np.float32, copy=True)
    rows = np.where(replace)[0]
    if rows.size:
        out_id[rows, amin[rows]] = act_id[rows, 0]
        out_lp[rows, amin[rows]] = act_lp[rows, 0]
    return out_id, out_lp


class TeacherCache:
    """缓存目录的只读访问层：index 加载、schema 断言、shard 惰性读取、转换。"""

    def __init__(self, cache_dir: str | Path, max_cached_shards: int = 2):
        self.dir = Path(cache_dir)
        cfg = json.loads((self.dir / "teacher_cfg.json").read_text())
        self.top_k: int = int(cfg["top_k"])
        self.model: str = cfg.get("model", "?")
        self._shards: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self._max_shards = max_cached_shards

        self._by_id: dict[str, list[SampleRef]] = {}
        with open(self.dir / "index.jsonl", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                ref = SampleRef(id=str(r["id"]), shard=r["shard"], row=int(r["row"]),
                                n_tokens=int(r["n_tokens"]), logit_rows=int(r["logit_rows"]),
                                top_k=int(r["top_k"]), file=r.get("file", ""),
                                scenario=r.get("scenario", ""))
                self._by_id.setdefault(ref.id, []).append(ref)
        n_bad = sum(1 for refs in self._by_id.values() for r in refs if r.top_k != self.top_k)
        if n_bad:
            raise ValueError(f"index 中 {n_bad} 条 top_k 与 teacher_cfg({self.top_k}) 不符")

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, sample_id: str) -> bool:
        return sample_id in self._by_id

    def records(self, sample_id: str) -> list[SampleRef]:
        """该 id 的全部缓存记录（语料 id 非唯一时可能 >1）。"""
        return self._by_id[sample_id]

    def ref(self, sample_id: str) -> SampleRef:
        """取末条记录（向后兼容；重复 id 请用 resolve() 内容级消歧）。"""
        return self._by_id[sample_id][-1]

    def resolve(self, sample_id: str, expected_ids: np.ndarray) -> SampleRef:
        """重复 id 下按 col0 token 流匹配真实样本（expected_ids = 重编码 ids[1:]）。

        语料 id 非唯一（如 opencodereasoning/tau-bench 每任务多块共享 id）时，
        按 id 取数会拿错块导致学生输入与教师 logits 错位。唯一 id 直接返回；
        重复 id 逐条比对 col0，恰好一条匹配才返回，否则报错。
        """
        recs = self._by_id[sample_id]
        if len(recs) == 1:
            return recs[0]
        expected = np.asarray(expected_ids, dtype=np.int32)
        matches = []
        for ref in recs:
            sh = self._shard(ref.shard)
            col0 = sh["token_ids"][ref.row:ref.row + ref.logit_rows, 0]
            if col0.shape[0] == expected.shape[0] and bool((col0 == expected).all()):
                matches.append(ref)
        if len(matches) == 1:
            return matches[0]
        raise ValueError(
            f"{sample_id}: 重复 id {len(recs)} 条中匹配到 {len(matches)} 条 "
            f"(expected_rows={expected.shape[0]})——内容级消歧失败")

    def plan_epoch(self, sample_ids) -> list[SampleRef]:
        """训练读取计划：缺失 id 报 KeyError（调用方须先过滤），按 (shard, row)
        聚簇排序，保证 shard 顺序读、每个 shard 只加载一次。

        注意：重复 id 的训练样本请先用 resolve()/resolved index 消歧到唯一记录，
        再据此规划读取顺序（本方法按 id 取末条仅适用唯一 id 集合）。
        """
        missing = [s for s in sample_ids if s not in self._by_id]
        if missing:
            raise KeyError(f"{len(missing)} 个样本不在缓存中，如 {missing[:3]}")
        refs = (self._by_id[s][-1] for s in sample_ids)
        return sorted(refs, key=lambda r: (r.shard, r.row))

    def _shard(self, name: str) -> dict[str, np.ndarray]:
        sh = self._shards.get(name)
        if sh is None:
            with np.load(self.dir / f"{name}.npz") as z:
                ids, lps = z["token_ids"], z["logits"]
            if ids.ndim != 2 or lps.ndim != 2 or ids.shape != lps.shape \
                    or ids.shape[1] != self.top_k + 1:
                raise ValueError(
                    f"{name}: 期望 [L,{self.top_k + 1}] 同形 token_ids/logits，"
                    f"实得 ids{ids.shape} logits{lps.shape}")
            if ids.dtype != np.int32 or lps.dtype != np.float32:
                raise ValueError(f"{name}: dtype 异常 ids={ids.dtype} logits={lps.dtype}")
            sh = {"token_ids": ids, "logits": lps}
            self._shards[name] = sh
            while len(self._shards) > self._max_shards:
                self._shards.popitem(last=False)   # 淘汰最旧
        else:
            self._shards.move_to_end(name)         # LRU 触碰
        return sh

    def fetch(self, sample_id: str,
              expected_ids: np.ndarray | None = None) -> FetchedSample:
        """取单样本：shard 行区间切片 + 列 0 合并转换。

        重复 id（语料 id 非唯一）必须传 expected_ids（重编码 ids[1:]）做内容级
        消歧，否则抛错——防止取到同 id 其他块的 logits 造成训练错位。
        """
        recs = self._by_id[sample_id]
        if len(recs) == 1:
            ref = recs[0]
        else:
            if expected_ids is None:
                raise ValueError(
                    f"{sample_id}: 缓存重复 id({len(recs)} 条)，须传 expected_ids 消歧")
            ref = self.resolve(sample_id, expected_ids)
        sh = self._shard(ref.shard)
        rows = slice(ref.row, ref.row + ref.logit_rows)
        ids = sh["token_ids"][rows]
        if ids.shape[0] != ref.logit_rows:   # 截断越界防护
            raise IndexError(f"{sample_id}: 行区间越过 shard 末尾")
        lps = sh["logits"][rows]
        t_ids, t_lps = merge_actual_topk(ids, lps)
        return FetchedSample(sample_id=sample_id, actual_ids=ids[:, 0].copy(),
                             topk_ids=t_ids, topk_lps=t_lps)

    def clear_shard_cache(self) -> None:
        self._shards.clear()
