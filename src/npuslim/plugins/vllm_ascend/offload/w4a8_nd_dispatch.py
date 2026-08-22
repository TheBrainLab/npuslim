"""W4A8 per-channel MoE ND dispatch patch (graph-mode offload, 2026-08-22).

背景（调查过程见 npuslim/docs/design/offload-graphmode-investigation.md §4/§5）:
- K2.6 类 W4A8（per-channel, group_size=0）MoE 前向恒 dispatch 到
  torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2 (moe_mlp.py:272)，
  其 910B aclnn 入口 (aclnnGroupedMatmulSwigluQuantWeightNzV2) 无条件把
  weight storage 绑定为 FRACTAL_NZ，拒绝 ND 权重 (EZ1001, 全 worker 复现)。
- VLLM_ASCEND_ENABLE_NZ=0 时 W4A8 权重保持 ND → 无论是否 offload / 图模式，
  profile_run 必崩。这是"K2.6 图模式 + offload"不可用的唯一缺口
  （W8A8/bf16 模型 NZ=0 + 图 + offload 已验证可用）。
- quant_apply_mlp 的 else ND 回退分支使用 torch_npu.npu_grouped_matmul
  (aclnnGroupedMatmulV5)。其 A8W4 场景官方支持 INT4 **ND/NZ** 权重
  (ops-transformer gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md)：
      y = ((x - 8) * w * scale + bias) * per_token_scale
      bias = 8 * sum_k(w) * scale   （离线辅助量，shape [E, N]）
  checkpoint 的 w13_scale_bias / w2_scale_bias 参数正是该 bias。
  数值已用硬件探针验证：ND 内核与文档公式一致（bf16 精度，rel 3.5e-3）。
- 本补丁在 NZ=0 时把 per-channel W4A8 路由到该 ND 回退分支，并把 swiglu
  改为 AIV 路径（npu_swiglu + npu_dynamic_quant）——triton-on-NPU 在
  aclgraph capture 下的可捕获性未验证，AIV op 为已知可捕获形态。
- scale 布局修正：vllm-ascend 把 per-channel W4A8 scale squeeze 成 2-D
  [E, N]（maybe_squeeze_per_channel_weight_scale），NZ 入口接受该形态，
  但 V5 A8W4 单 tensor tiling 按 [E, quantGroupNum, N] 解析
  （grouped_matmul_tiling.cpp A8W4Tiling: n=Dim(2), quantGroupNum=Dim(1)）
  → 2-D 被误读为 quantGroupNum=N → tiling 失败 "k should be divisible by
  quantGroupNum"（b3 首轮实测）。补丁在路由到 ND 分支时 unsqueeze 回
  [E, 1, N]（纯 view，图内安全）。

门控（全部满足才生效）:
  1. NPUSLIM_W4A8_ND_DISPATCH != "0"   （kill switch，默认开）
  2. VLLM_ASCEND_ENABLE_NZ == "0"      （权重确为 ND）

不受影响: W8A8 模型（quant_apply_mlp branch 1）、NZ=1 的 K2.6（生产路径
原样）、无 per-channel W4A8 scale_bias 的模型（wrapper 为 no-op）。

放置说明: 位于 offload/ 目录（随 trunk 插件一起分发/发现），但生效条件与 offload
无关——K2.6 类 W4A8 模型在 NZ=0 下的任何部署（含无 offload）都需要本补丁，
否则 profile_run 崩溃（investigation §4.2）。
"""

from __future__ import annotations

import os
from typing import Any

from npuslim.plugins.logging import patch_logger
from npuslim.plugins.registry import register_patch


def _nd_dispatch_active() -> bool:
    if os.environ.get("NPUSLIM_W4A8_ND_DISPATCH", "1") == "0":
        return False
    return os.environ.get("VLLM_ASCEND_ENABLE_NZ", "1") == "0"


@register_patch(target="vllm_ascend.ops.fused_moe.moe_mlp")
def patch_w4a8_nd_dispatch(module: Any) -> None:
    """Route per-channel W4A8 MoE to the ND fallback branch when NZ=0.

    Forces ``use_w4a8_per_channel_gmm_swiglu=False`` and ``fusion=False`` for
    calls that carry ``w1_scale_bias`` (the per-channel W4A8 branch marker),
    so quant_apply_mlp's dispatch falls through to the else branch:
        npu_grouped_matmul (V5 A8W4, INT4 ND)
        + npu_swiglu + npu_dynamic_quant   (AIV; HAS_TRITON forced off)
        + npu_grouped_matmul (V5 A8W4, INT4 ND)
    All ND ops, graph-capturable per the ND copy_ / AIV capture evidence.
    """
    if not _nd_dispatch_active():
        patch_logger.info(
            "[W4A8ND] patch not applied "
            f"(NPUSLIM_W4A8_ND_DISPATCH={os.environ.get('NPUSLIM_W4A8_ND_DISPATCH', '1')}, "
            f"VLLM_ASCEND_ENABLE_NZ={os.environ.get('VLLM_ASCEND_ENABLE_NZ', '1')})"
        )
        return

    original = module.quant_apply_mlp

    # Use the AIV swiglu path inside the else branch (graph-capturable);
    # the triton-on-NPU swiglu_quant capturability under aclgraph is unverified.
    module.HAS_TRITON = False

    def patched_quant_apply_mlp(hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("w1_scale_bias") is not None:
            # per-channel W4A8 branch marker: force ND fallback dispatch
            kwargs["use_w4a8_per_channel_gmm_swiglu"] = False
            kwargs["fusion"] = False
            # V5 A8W4 single-tensor tiling reads the scale layout as
            # [E, quantGroupNum, N] (grouped_matmul_tiling.cpp A8W4Tiling:
            # n=Dim(2), quantGroupNum=Dim(1)), but vllm-ascend squeezes the
            # per-channel W4A8 scale to 2-D [E, N]
            # (maybe_squeeze_per_channel_weight_scale) — that form is only
            # accepted by the NZ entries. Restore the 3-D layout for the ND
            # path. unsqueeze is a view (metadata only) — graph-capture safe.
            for key in ("w1_scale", "w2_scale"):
                scales = kwargs.get(key)
                if (
                    isinstance(scales, (list, tuple))
                    and len(scales) == 1
                    and scales[0] is not None
                    and scales[0].dim() == 2
                ):
                    kwargs[key] = [scales[0].unsqueeze(1)]
        return original(hidden_states, *args, **kwargs)

    module.quant_apply_mlp = patched_quant_apply_mlp

    patch_logger.success(
        "[W4A8ND] ND dispatch ACTIVE: per-channel W4A8 MoE -> "
        "V5 A8W4 fallback (gmm1 + AIV swiglu + gmm2), triton swiglu off"
    )
