"""A7 Step B：GPTQ 旋转子类（R1 折叠 + hook 注入）—— 真实规模 AWQ/GPTQ 对照（调研文档 §6.1 Step 4-6）。

方案 B：独立子类 `GPTQRotationAlgorithm`，继承 `GPTQAlgorithm`，**生产代码零改动**。
默认 `rotation=None` → 行为与 `GPTQAlgorithm` 完全一致（不影响既有量化产线）。

折叠（行向量约定，y = x @ W.T，R 正交）—— 消费 R1 旋转激活的权重**列折** W ← W @ R1.T：
- `self_attn.q_proj / k_proj / v_proj`（喂 post input_layernorm 旋转激活）
- `mlp.gate`（router，喂 post_attention_layernorm 旋转激活）
- `mlp.experts.gate_up_proj`（3D [E, 2I, H]，每专家同 R1）
- **不折**：`o_proj` / `down_proj`（其输入 = attn 输出 / swiglu 输出，未被旋转）

hook（norm 后、量化 matmul 边界，与部署在线 kernel 作用位置一致）：
- `input_layernorm` 输出 × R1.T（喂 q/k/v_proj）
- `post_attention_layernorm` 输出 × R1.T（喂 router + gate_up）

时序（§6.1 Step 2/4）：
- 折叠必须在统计采集之前（AWQ x_abs_mean 基于折叠后权重产生的激活）
- 统计 `_collect_statistics` 与前向接力 `_forward_layer_outputs` 期间 hook 必须生效
  （折叠 + hook = fp 恒等；缺一即静默错误——折叠未 hook 会改变函数）
- 只对**被量化的层**折叠+hook；完全跳过的层原样通过（正确 fp 接力）
- 打包/保存/部署链路零改动（产物不可部署，需在线 kernel §2.5）
- **快照同步**：折叠原地改的是模型内存权重，而 chunk 张量快照是加载副本；
  量化模块的快照会被 GPTQ handler 用打包结果替换（基于折叠后权重，正确），
  但 FLOAT 透传（router `mlp.gate`）保留加载副本 → 产物中出现未折叠 router。
  `process_chunk` 末尾把被折叠层的 FLOAT 快照同步回折叠值（Qwen3-30B 实测踩坑：
  router 未同步导致 ROT 臂 ppl 6111 vs NOROT 11.95）。

R1 矩阵：`create_rot`（qad/rotation.py，与 msmodelslim 逐位一致），seed 确定性。
折叠与旋转均在 fp32 计算再 cast 回 bf16，把 fp 等价性误差压到 bf16 舍入下限。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from loguru import logger

from npuslim.algorithms.quantization.gptq.gptq_algo import GPTQAlgorithm
from npuslim.core import AlgorithmRegistry
from npuslim.qad.rotation import QuaRotMode, create_rot


# ---------------- 可独立测试的折叠 / hook 原语 ----------------

def fold_layer_for_rotation(layer_module: nn.Module, r1: torch.Tensor) -> None:
    """对 Qwen3MoE 层做 R1 列折（原地 copy_，保持 3D expert slice view 有效）。

    Args:
        layer_module: Qwen3MoeDecoderLayer
        r1: [H, H] 正交旋转矩阵（fp32）
    """
    r1_t = r1.T.float()
    with torch.no_grad():
        attn = layer_module.self_attn
        for name in ("q_proj", "k_proj", "v_proj"):
            w = getattr(attn, name).weight
            if w.shape[-1] == r1.shape[-1]:
                w.data.copy_((w.float() @ r1_t).to(w.dtype))
        gate = layer_module.mlp.gate.weight  # router
        if gate.shape[-1] == r1.shape[-1]:
            gate.data.copy_((gate.float() @ r1_t).to(gate.dtype))
        experts = layer_module.mlp.experts
        w3d = getattr(experts, "gate_up_proj", None)  # 3D [E, 2I, H]
        if w3d is not None and w3d.ndim == 3 and w3d.shape[-1] == r1.shape[-1]:
            w3d.data.copy_((w3d.float() @ r1_t).to(w3d.dtype))


def register_rotation_hooks(layer_module: nn.Module, r1: torch.Tensor) -> List[torch.utils.hooks.RemovableHandle]:
    """注册 R1 旋转 forward hook（norm 后激活 × R1.T），返回 handle 列表（调用方负责 remove）。

    折叠 + 本 hook 同时生效才是 fp 恒等；单独启用任一即改变模型函数。
    """
    r1_t = r1.T.float()

    def make_hook():
        def hook(module, inp, out):
            return (out.float() @ r1_t).to(out.dtype)
        return hook

    handles = [
        layer_module.input_layernorm.register_forward_hook(make_hook()),
        layer_module.post_attention_layernorm.register_forward_hook(make_hook()),
    ]
    return handles


# ---------------- 算法子类 ----------------

@AlgorithmRegistry.register("GPTQRotation", aliases=["gptq_rotation", "gptq_rot"])
class GPTQRotationAlgorithm(GPTQAlgorithm):
    """GPTQ + 可选的 R1 Hadamard 旋转（AWQ/GPTQ 前端叠加，真实规模对照）。

    rotation 配置（algorithm.rotation，默认 None 即关闭）：
        mode: hadamard | block_hadamard_shifted
        block_size: -1(全维) | 128 | 256
        seed: R1 生成种子（与部署 seed 约定一致，§2.5）
    """

    _TAG = "GPTQRot"

    def __init__(self, rotation: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        self._rotation_cfg: Dict[str, Any] = dict(rotation or {})
        self._r1: Optional[torch.Tensor] = None
        self._layer_folded: set[str] = set()
        self._current_chunk = None

    @property
    def rotation_enabled(self) -> bool:
        return bool(self._rotation_cfg)

    def _ensure_r1(self) -> None:
        """惰性创建 R1（首次折叠时，此时 _runtime_device 已由 process_chunk 解析）。"""
        if self._r1 is not None or not self._rotation_cfg:
            return
        hidden = getattr(getattr(self._runtime_model, "config", None), "hidden_size", None)
        if hidden is None:
            raise ValueError("[GPTQRot] cannot resolve hidden_size for rotation")
        mode_name = str(self._rotation_cfg.get("mode", "hadamard")).lower()
        try:
            mode = QuaRotMode(mode_name)
        except ValueError:
            raise ValueError(f"[GPTQRot] unsupported rotation mode: {mode_name}")
        block_size = int(self._rotation_cfg.get("block_size", -1))
        seed = int(self._rotation_cfg.get("seed", 2026))
        r1 = create_rot(mode, int(hidden), block_size=block_size, seed=seed,
                        dtype=torch.float32, device=torch.device("cpu"))
        device = self._runtime_device if self._runtime_device is not None else torch.device("cpu")
        self._r1 = r1.to(device, dtype=torch.float32)
        logger.info(f"[GPTQRot] R1 旋转启用: mode={mode.value}, block_size={block_size}, "
                    f"seed={seed}, size={r1.shape}, device={device}")
        self._layer_folded.clear()

    # ---- 折叠 + hook 注入（统计前向 / 接力前向两处，保持一致） ----

    def _ensure_layer_rotated(self, layer_module: nn.Module, layer_name: str) -> None:
        if layer_name in self._layer_folded:
            return
        self._ensure_r1()
        fold_layer_for_rotation(layer_module, self._r1)
        self._layer_folded.add(layer_name)

    # ---- 快照同步（防止 FLOAT 透传张量以未折叠态落盘） ----
    #
    # 根因（Qwen3-30B 实测踩坑）：折叠原地改的是模型内存权重，而 chunk 张量快照
    # 是加载副本；量化模块的快照会被 GPTQ handler 用打包结果替换（基于折叠后
    # 权重，正确），但 FLOAT 透传（router `mlp.gate.weight`）保留加载副本 →
    # 产物中出现未折叠 router → ROT 臂部署/评测时路由全错（ppl 6111 vs 11.95）。
    #
    # 同步时机必须在折叠当下（_collect_statistics 内）：每层处理结束后模块权重
    # 被 unassign 置为 meta，chunk 末尾无法从模块读取。同步值直接从 chunk 快照
    # 重算（snap @ R1^T，fp32→原 dtype），与 in-memory 折叠语义一致，且不依赖
    # 模块生命周期/设备。量化模块同步后会被打包结果覆盖，无副作用。

    def process_chunk(self, chunk):
        """覆写：暂存 chunk 引用，供折叠时同步 FLOAT 快照。"""
        self._current_chunk = chunk
        try:
            return super().process_chunk(chunk)
        finally:
            self._current_chunk = None

    def _sync_layer_float_snapshots(self, layer_name: str) -> None:
        """把 layer_name 层被折叠的 FLOAT 透传快照同步为折叠值。"""
        chunk = self._current_chunk
        if chunk is None or self._r1 is None:
            return
        r1_t = self._r1.T.float()
        candidates = [
            f"{layer_name}.mlp.gate.weight",  # router（实测中招）
            f"{layer_name}.mlp.experts.gate_up_proj",  # 融合 3D 快照（fuse 后无 .weight 后缀）
        ]
        for proj in ("q_proj", "k_proj", "v_proj"):
            candidates.append(f"{layer_name}.self_attn.{proj}.weight")
        synced = 0
        for name in candidates:
            snap = chunk.get_tensor(name)
            if snap is None or not torch.is_floating_point(snap):
                continue
            folded = (snap.float() @ r1_t.to(snap.device)).to(snap.dtype)
            chunk.update_tensor(name, folded)
            synced += 1
        if synced:
            logger.info(f"[GPTQRot] {layer_name}: FLOAT 透传快照已同步折叠值 ({synced} 个)")

    def _collect_statistics(self, layer_module, handlers, *, layer_name, chunk_index):
        """统计前向：折叠 + R1 hook 必须在此生效（x_abs_mean 基于折叠后激活）。"""
        if self.rotation_enabled:
            self._ensure_layer_rotated(layer_module, layer_name)
            self._sync_layer_float_snapshots(layer_name)
            handles = register_rotation_hooks(layer_module, self._r1)
            try:
                return super()._collect_statistics(
                    layer_module, handlers, layer_name=layer_name, chunk_index=chunk_index
                )
            finally:
                for h in handles:
                    h.remove()
        return super()._collect_statistics(
            layer_module, handlers, layer_name=layer_name, chunk_index=chunk_index
        )

    def _forward_layer_outputs(self, layer_module, *, layer_name, chunk_index):
        """接力前向：仅对已折叠层挂 hook（折叠+hook=fp 恒等；未折叠层原样通过）。"""
        if self.rotation_enabled and layer_name in self._layer_folded:
            handles = register_rotation_hooks(layer_module, self._r1)
            try:
                return super()._forward_layer_outputs(
                    layer_module, layer_name=layer_name, chunk_index=chunk_index
                )
            finally:
                for h in handles:
                    h.remove()
        return super()._forward_layer_outputs(
            layer_module, layer_name=layer_name, chunk_index=chunk_index
        )
