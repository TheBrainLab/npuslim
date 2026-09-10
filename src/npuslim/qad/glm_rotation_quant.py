"""GPTQ + GLM-5.2 残差流旋转折叠管线（E3-4，Layer 1 复刻配方）。

残差流折叠（msmodelslim model/glm_5/model_adapter.py + quarot.py，quarot 处理器）：
  - embed_tokens 输出旋转（pre_run，right rot）
  - 每层：norm γ 融合（γ→1，折进全部消费者含 DSA indexer）+ 旋转折叠
    （残差流 right rot / o_proj、down 左旋 / MLA 三套子空间旋转）
  - model.norm → lm_head（post_run：γ 融合 + lm_head right rot）

折叠后网络函数恒等（fp 等价闸门见 tests/qad/test_glm_rotation.py），统计前向/接力
前向在旋转空间自然进行，**无需在线 hook、产物可直接部署**（与 Qwen3 GPTQRot 的
R1-hook 式「产物不可部署」不同）。

顺序（与 modelslim best_practice 一致）：
  旋转折叠（_collect_statistics 内、统计前）
  → AWQ/SQ（_apply_scale_search，GPTQAlgorithm 既有逻辑，在折叠后权重上执行）
  → RTN/SSZ/GPTQ 量化 + 打包

快照同步（A7 router 陷阱泛化）：FLOAT 透传张量（norms/router/kv_b_proj/indexer
wk/weights_proj）的 chunk 快照必须同步为折叠值；量化张量（q_a/kv_a/q_b/o_proj/
wq_b/专家/shared/dense）的 float 快照由打包结果 pop+替换，无需同步。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from loguru import logger

from npuslim.algorithms.quantization.gptq.gptq_algo import GPTQAlgorithm
from npuslim.algorithms.quantization.hessian import _get_child_module
from npuslim.core import AlgorithmRegistry
from npuslim.qad.glm_rotation import (
    fold_glm52_layer,
    get_glm52_rotate_maps,
)


@AlgorithmRegistry.register("GPTQGlmRotation", aliases=["gptq_glm_rotation", "gptq_glmrot"])
class GPTQGlmRotationAlgorithm(GPTQAlgorithm):
    """GPTQ + GLM-5.2 残差流旋转折叠（modelslim 复刻 Layer 1 配方）。

    配置（algorithm.glm_rotation）：
        block_size: -1（全维 Kronecker，默认）
        seed: 2026（R 矩阵生成种子，须与部署侧一致）
    其余参数（wbits/act_bits/layer_routing/scale_search/smooth_quant/expert_mode/
    ssz_step）与 GPTQAlgorithm 相同。
    """

    _TAG = "GPTQGlmRot"

    def __init__(self, glm_rotation: Optional[Dict] = None, **kwargs):
        super().__init__(**kwargs)
        self._glm_rot_cfg = dict(glm_rotation or {})
        self._rot_matrices: Optional[Dict[str, torch.Tensor]] = None
        self._rot_params: Dict[str, int] = {}
        self._current_chunk = None
        self._embed_folded = False
        self._final_folded = False

    # ---------------- 配置 / 构建 ----------------

    @property
    def glm_rotation_enabled(self) -> bool:
        return bool(self._glm_rot_cfg)

    def _log_start_params(self) -> None:
        super()._log_start_params()
        if self._glm_rot_cfg:
            logger.info(f"[{self._TAG}] glm_rotation enabled: {self._glm_rot_cfg}")

    def _build_rotation(self) -> None:
        cfg = self._runtime_model.config
        block_size = int(self._glm_rot_cfg.get("block_size", -1))
        seed = int(self._glm_rot_cfg.get("seed", 2026))
        self._rot_params = dict(
            num_layers=int(cfg.num_hidden_layers),
            first_k_dense_replace=int(cfg.first_k_dense_replace),
            qk_nope_head_dim=int(getattr(cfg, "qk_nope_head_dim", 192)),
            qk_rope_head_dim=int(getattr(cfg, "qk_rope_head_dim", 64)),
        )
        _, _, rm = get_glm52_rotate_maps(
            hidden=int(cfg.hidden_size),
            q_lora_rank=int(cfg.q_lora_rank),
            kv_lora_rank=int(cfg.kv_lora_rank),
            v_head_dim=int(cfg.v_head_dim),
            block_size=block_size,
            seed=seed,
            **{k: self._rot_params[k] for k in ("num_layers", "first_k_dense_replace")},
        )
        self._rot_matrices = rm
        logger.info(
            f"[{self._TAG}] GLM residual-stream rotation built: "
            f"hidden={cfg.hidden_size}, q_lora={cfg.q_lora_rank}, "
            f"kv_lora={cfg.kv_lora_rank}, v_head={cfg.v_head_dim}, "
            f"block_size={block_size}, seed={seed}, "
            f"layers={self._rot_params['num_layers']}, "
            f"dense_first={self._rot_params['first_k_dense_replace']}"
        )

    def _layer_index(self, layer_name: str) -> int:
        return int(layer_name.rsplit(".", 1)[1])

    # ---------------- 生命周期 ----------------

    def on_start(self) -> None:
        super().on_start()
        if self.glm_rotation_enabled:
            self._build_rotation()

    def process_chunk(self, chunk):
        self._current_chunk = chunk
        try:
            if self.glm_rotation_enabled and chunk.post_modules and not self._final_folded:
                self._fold_post_modules(chunk)
            return super().process_chunk(chunk)
        finally:
            self._current_chunk = None

    # ---------------- 折叠 ----------------

    def _fold_pre_modules(self, chunk) -> None:
        """chunk 0：embed_tokens 输出旋转（right rot）+ chunk/运行时模型双写。

        调用时机：_capture_initial_inputs 前（embed 已 assign 进运行时模型，
        校准输入采集将使用折叠后 embed → 残差流自始活在 R 空间）。
        """
        rot = self._rot_matrices["rot"]
        for module in chunk.pre_modules:
            if "embed_tokens" not in module.name:
                continue
            t = module.tensors.get("weight")
            if t is None:
                continue
            w = t.float()
            folded = (w @ rot.to(device=w.device, dtype=torch.float32)).to(t.dtype)
            chunk.update_tensor(f"{module.name}.weight", folded)
            embed_mod = _get_child_module(self._runtime_model, module.name)
            if embed_mod is not None and hasattr(embed_mod, "weight"):
                # embed 可能已被 _assign_runtime_tensors 迁移到运行时设备（NPU），
                # folded 基于 chunk 张量（CPU）计算 → 必须迁到模块当前设备
                embed_mod.weight.data = folded.to(
                    device=embed_mod.weight.device, dtype=embed_mod.weight.dtype
                )
            self._embed_folded = True
            logger.info(f"[{self._TAG}] embed_tokens folded (right rot {tuple(rot.shape)})")
            return
        logger.warning(f"[{self._TAG}] embed_tokens not found in chunk.pre_modules")

    def _capture_initial_inputs(self, chunk):
        if self.glm_rotation_enabled and not self._embed_folded and self._inps is None:
            self._fold_pre_modules(chunk)
        return super()._capture_initial_inputs(chunk)

    def _collect_statistics(self, layer_module, handlers, *, layer_name, chunk_index):
        if self.glm_rotation_enabled and self._rot_matrices is not None:
            idx = self._layer_index(layer_name)
            if idx < self._rot_params["num_layers"]:
                fold_glm52_layer(layer_module, idx, self._rot_matrices, **self._rot_params)
                self._sync_layer_float_snapshots(layer_name)
            else:
                logger.warning(
                    f"[{self._TAG}] layer {idx} >= num_hidden_layers "
                    f"({self._rot_params['num_layers']}), fold skipped (MTP/R1 scope)"
                )
        return super()._collect_statistics(
            layer_module, handlers, layer_name=layer_name, chunk_index=chunk_index
        )

    def _sync_layer_float_snapshots(self, layer_name: str) -> None:
        """FLOAT 透传张量 chunk 快照 ← 折叠后的运行时模型值。

        量化张量（q_a/kv_a/q_b/o_proj/wq_b/专家 3D/shared/dense）的 float 快照
        由打包 pop+替换，无需在此同步；此处仅覆盖保持 FLOAT 的：
        norms（γ→1，SQ 阶段可能再同步为 1/s）、router gate、kv_b_proj、
        indexer.wk/weights_proj。
        """
        chunk = self._current_chunk
        if chunk is None:
            return
        layer_module = _get_child_module(self._runtime_model, layer_name)
        if layer_module is None:
            return
        candidates = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_a_layernorm.weight",
            "self_attn.kv_a_layernorm.weight",
            "mlp.gate.weight",
            "self_attn.kv_b_proj.weight",
            "self_attn.indexer.wk.weight",
            "self_attn.indexer.weights_proj.weight",
        ]
        synced = 0
        for rel in candidates:
            full = f"{layer_name}.{rel}"
            snap = chunk.get_tensor(full)
            if snap is None:
                continue
            # 遍历到「模块」为止（剥掉 .weight 后缀），再读 mod.weight
            mod_path = rel[: -len(".weight")] if rel.endswith(".weight") else rel
            mod = layer_module
            for part in mod_path.split("."):
                mod = getattr(mod, part, None)
                if mod is None:
                    break
            if mod is None or not hasattr(mod, "weight"):
                continue
            val = mod.weight.detach().to(device=snap.device, dtype=snap.dtype)
            chunk.update_tensor(full, val)
            synced += 1
        if synced:
            logger.info(f"[{self._TAG}] {layer_name}: FLOAT 快照同步 {synced} 个")

    def _fold_post_modules(self, chunk) -> None:
        """末 chunk：model.norm → lm_head（γ 融合 + lm_head right rot），张量级折叠。

        model.norm / lm_head 不参与校准前向（_capture_initial_inputs 首层即停，
        逐层前向只跑 decoder layer），直接折叠 chunk 张量即可。
        """
        norm_t = chunk.get_tensor("model.norm.weight")
        lm_t = chunk.get_tensor("lm_head.weight")
        if norm_t is None or lm_t is None:
            logger.warning(
                f"[{self._TAG}] model.norm/lm_head not found in post_modules, "
                f"final fold skipped"
            )
            return
        rot = self._rot_matrices["rot"]
        gamma = norm_t.float()
        lm = lm_t.float()
        # 1) γ 融合：lm ← lm * γ（列），γ → 1
        lm_fused = (lm * gamma.view(1, -1)).to(lm_t.dtype)
        # 2) lm_head right rot：残差流（R 空间）→ 吸收
        lm_rot = (lm_fused.float() @ rot.to(device=lm_fused.device, dtype=torch.float32)).to(lm_t.dtype)
        chunk.update_tensor("model.norm.weight", torch.ones_like(norm_t))
        chunk.update_tensor("lm_head.weight", lm_rot)
        self._final_folded = True
        logger.info("[{tag}] model.norm γ→1, lm_head fused+rotated".format(tag=self._TAG))
