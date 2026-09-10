"""GLM-5.2 残差流旋转折叠（E3）—— 移植 msmodelslim quarot 离线折叠。

参考实现：
    msmodelslim/model/glm_5/quarot.py           (get_rotate_map / get_ln_fuse_map)
    msmodelslim/processor/quarot/common/quarot_utils.py (rotate_linear / fuse_ln_linear)
    msmodelslim/processor/quarot/offline_quarot/quarot.py (QuaRotProcessor 编排)

数学（行向量约定 y = x @ W.T，旋转在线形式 x @ R.T）：
  - right_rot[W] = R：该线性**输入**在旋转空间，折 W ← W @ R 吸收（等价 W @ R.T 折叠）
  - left_rot[W]  = R：该线性**输出**被旋转，折 W ← R.T @ W（+ bias ← R.T @ bias）
  - fuse_ln_linear：norm γ 折进消费者线性输入列（W ← W * γ），norm γ 置 1（RMS 归一项保留）

残差流旋转版式（w4a8c8 生产所用，§1.6.2 证据#3）：
  - embed_tokens 输出旋转（pre_run）；o_proj / down_proj 输出侧旋转（left_rot）
    —— 残差流全程活在旋转空间，引擎零改动
  - MLA 空间旋转：q_b_proj (rot_b_proj, q_lora_rank=2048, BLOCK_HADAMARD_SHIFTED)、
    kv_b_proj (rot_kv_b_proj, kv_lora_rank=512)、o_proj 输入 v 空间 (rot_uv, v_head_dim=256)

折叠时序（§2.4 E3 步骤 3）：γ 融合 → 旋转 → AWQ/SQ → GPTQ → 打包。
本模块只负责「γ 融合 + 旋转」；AWQ/SQ/GPTQ 由量化管线在旋转后权重上继续。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from npuslim.qad.rotation import QuaRotMode, create_rot


# ---------------- 折叠原语（移植 quarot_utils.py） ----------------

def rotate_linear(linear: nn.Module, rot: torch.Tensor, right_rotate: bool = True) -> None:
    """对 2D 线性层权重应用旋转折叠（原地改写 weight / bias）。

    Args:
        linear: 带 .weight（2D [out, in]）的线性模块
        rot: [d, d] 正交矩阵（或 list/tuple → block_diag）
        right_rotate: True 折输入侧 W ← W @ rot；False 折输出侧 W ← rot.T @ W（bias 同转）
    """
    if isinstance(rot, (list, tuple)):
        rot = torch.block_diag(*rot)
    dtype = linear.weight.data.dtype
    device = linear.weight.device
    weight_data = linear.weight.data.to(device=device, dtype=torch.float32)
    dim_size = weight_data.shape[1] if right_rotate else weight_data.shape[0]
    if dim_size != rot.shape[0]:
        block_num = dim_size // rot.shape[0]
        if dim_size % rot.shape[0] != 0:
            raise ValueError(
                f"rotate matrix dim {rot.shape[0]} must be a divisor of weight dim {dim_size}"
            )
        rot = torch.block_diag(*([rot] * block_num))
    if right_rotate:
        linear.weight.data = torch.matmul(weight_data, rot.to(weight_data)).to(dtype=dtype)
    else:
        linear.weight.data = torch.matmul(rot.T.to(weight_data), weight_data).to(dtype=dtype)
        if getattr(linear, "bias", None) is not None:
            bias_data = linear.bias.data.to(device=device, dtype=torch.float32)
            linear.bias.data = torch.matmul(rot.T.to(weight_data), bias_data).to(dtype=dtype)


def rotate_weight_3d(weight: nn.Parameter, rot: torch.Tensor, right_rotate: bool = True) -> None:
    """对 3D 融合 MoE 参数 [E, out, in] 逐专家应用旋转折叠（原地 copy_）。

    right_rotate: W ← W @ rot（输入侧，作用于 in 维）
    left_rotate:  W ← rot.T @ W（输出侧，作用于 out 维）
    """
    if isinstance(rot, (list, tuple)):
        rot = torch.block_diag(*rot)
    dtype = weight.data.dtype
    device = weight.data.device
    w = weight.data.float()
    dim_size = w.shape[2] if right_rotate else w.shape[1]
    if dim_size != rot.shape[0]:
        block_num = dim_size // rot.shape[0]
        if dim_size % rot.shape[0] != 0:
            raise ValueError(
                f"rotate matrix dim {rot.shape[0]} must be a divisor of weight dim {dim_size}"
            )
        rot = torch.block_diag(*([rot] * block_num))
    rot = rot.to(device)
    if right_rotate:
        w = torch.matmul(w, rot)  # [E, out, in] @ [in, in]
    else:
        w = torch.matmul(rot.T, w)  # [out, out] @ [E, out, in]
    weight.data.copy_(w.to(dtype=dtype))


def fuse_ln_linear(layernorms: List[nn.Module], linear_layers: List[nn.Module]) -> None:
    """把 norm 增益 γ 折进消费者线性输入列，norm γ 置 1（RMS 归一保留）。

    W_linear ← W_linear * γ（per-input-channel）；layernorm.weight ← 1。
    """
    ln_data = []
    for ln in layernorms:
        if ln.weight.dim() != 1:
            raise ValueError("only RMSNorm-style (1D gain) layernorm supported")
        ln_data.append(ln.weight.data)
    ln_weight = torch.concat(ln_data, dim=0) if len(ln_data) > 1 else ln_data[0]
    ln_weight = ln_weight.to(dtype=torch.float32)
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        current_weight = linear.weight.data.to(dtype=torch.float32)
        if ln_weight.shape[0] != current_weight.shape[1]:
            raise ValueError(
                f"layernorm weight dim {ln_weight.shape[0]} != linear input dim "
                f"{current_weight.shape[1]}"
            )
        linear.weight.data = (
            current_weight * ln_weight.to(device=current_weight.device)
        ).to(linear_dtype)
    for ln in layernorms:
        ln.weight.data.fill_(1.0)


# ---------------- 旋转映射（移植 model/glm_5/quarot.py） ----------------

@dataclass
class RotateCommand:
    target: str            # 模块相对路径（层内，如 "self_attn.q_a_proj"）
    rot: torch.Tensor
    right_rotate: bool


def get_glm52_rotate_maps(
    hidden: int = 6144,
    q_lora_rank: int = 2048,
    kv_lora_rank: int = 512,
    v_head_dim: int = 256,
    num_layers: int = 78,
    first_k_dense_replace: int = 3,
    block_size: int = -1,
    seed: int = 2026,
) -> Tuple[List[RotateCommand], List[RotateCommand], Dict[str, torch.Tensor]]:
    """构造 GLM-5.2 旋转命令（pre_run / 主流程），逐层目标名以层内相对路径给出。

    返回 (pre_run_commands, layer_commands, rotate_matrices)：
      - pre_run_commands: embed_tokens 输出旋转（right_rot=rot）
      - layer_commands: 每层通用的命令序列（target 为层内相对路径）
      - rotate_matrices: {'rot','rot_b_proj','rot_uv','rot_kv_b_proj'}
    """
    rot = create_rot(QuaRotMode.HADAMARD, int(hidden), block_size=block_size, seed=seed)
    rot_b_proj = create_rot(
        QuaRotMode.BLOCK_HADAMARD_SHIFTED, int(q_lora_rank), block_size=block_size, seed=seed
    )
    rot_uv = create_rot(QuaRotMode.HADAMARD, int(v_head_dim), block_size=block_size, seed=seed)
    rot_kv_b_proj = create_rot(
        QuaRotMode.HADAMARD, int(kv_lora_rank), block_size=block_size, seed=seed
    )

    pre_run = [RotateCommand("embed_tokens", rot, right_rotate=True)]

    cmds: List[RotateCommand] = []
    # 残差流（rot）：q_a/kv_a 输入侧；o_proj 输出侧；dense/专家/shared gate_up 输入侧、down 输出侧；router 输入侧
    for i in range(num_layers):
        cmds.append(RotateCommand(f"layers.{i}.self_attn.q_a_proj", rot, right_rotate=True))
        cmds.append(RotateCommand(f"layers.{i}.self_attn.kv_a_proj_with_mqa", rot, right_rotate=True))
        cmds.append(RotateCommand(f"layers.{i}.self_attn.o_proj", rot, right_rotate=False))
        if i < first_k_dense_replace:
            cmds.append(RotateCommand(f"layers.{i}.mlp.gate_proj", rot, right_rotate=True))
            cmds.append(RotateCommand(f"layers.{i}.mlp.up_proj", rot, right_rotate=True))
            cmds.append(RotateCommand(f"layers.{i}.mlp.down_proj", rot, right_rotate=False))
        else:
            cmds.append(RotateCommand(f"layers.{i}.mlp.experts.gate_up_proj", rot, right_rotate=True))
            cmds.append(RotateCommand(f"layers.{i}.mlp.experts.down_proj", rot, right_rotate=False))
            cmds.append(RotateCommand(f"layers.{i}.mlp.shared_experts.gate_proj", rot, right_rotate=True))
            cmds.append(RotateCommand(f"layers.{i}.mlp.shared_experts.up_proj", rot, right_rotate=True))
            cmds.append(RotateCommand(f"layers.{i}.mlp.shared_experts.down_proj", rot, right_rotate=False))
            cmds.append(RotateCommand(f"layers.{i}.mlp.gate", rot, right_rotate=True))
    # MTP 层输入（与主层一致，层 77 处理完后接 MTP）
    cmds.append(RotateCommand("layers.77.embed_tokens", rot, right_rotate=True))
    # lm_head（model.norm 输出在旋转空间）
    cmds.append(RotateCommand("lm_head", rot, right_rotate=True))
    # MLA 空间：q_a 输出（lora 空间）左旋，q_b 输入右旋
    for i in range(num_layers):
        cmds.append(RotateCommand(f"layers.{i}.self_attn.q_a_proj", rot_b_proj, right_rotate=False))
        cmds.append(RotateCommand(f"layers.{i}.self_attn.q_b_proj", rot_b_proj, right_rotate=True))
    # MLA 空间：kv_b 输出 v 部分旋转（[eye(qk_nope), rot_uv] 块对角），o_proj 输入右旋
    for i in range(num_layers):
        cmds.append(RotateCommand(
            f"layers.{i}.self_attn.kv_b_proj",
            [torch.eye(192, dtype=rot_uv.dtype, device=rot_uv.device), rot_uv],
            right_rotate=False,
        ))
        cmds.append(RotateCommand(f"layers.{i}.self_attn.o_proj", rot_uv, right_rotate=True))
    # MLA 空间：kv_a 输出 c 部分旋转（[rot_kv_b, eye(qk_rope)] 块对角），kv_b 输入右旋
    for i in range(num_layers):
        cmds.append(RotateCommand(
            f"layers.{i}.self_attn.kv_a_proj_with_mqa",
            [rot_kv_b_proj, torch.eye(64, dtype=rot_kv_b_proj.dtype, device=rot_kv_b_proj.device)],
            right_rotate=False,
        ))
        cmds.append(RotateCommand(f"layers.{i}.self_attn.kv_b_proj", rot_kv_b_proj, right_rotate=True))

    return pre_run, cmds, {"rot": rot, "rot_b_proj": rot_b_proj, "rot_uv": rot_uv,
                            "rot_kv_b_proj": rot_kv_b_proj}


def _layer_commands(
    layer_index: int,
    rm: Dict[str, torch.Tensor],
    num_layers: int,
    first_k_dense_replace: int,
    qk_nope_head_dim: int = 192,
    qk_rope_head_dim: int = 64,
) -> List[RotateCommand]:
    """返回某一 decoder 层的旋转命令（target 为层内相对路径，基于给定旋转矩阵）。"""
    rot = rm["rot"]
    rot_b_proj = rm["rot_b_proj"]
    rot_uv = rm["rot_uv"]
    rot_kv_b_proj = rm["rot_kv_b_proj"]
    cmds: List[RotateCommand] = []
    cmds.append(RotateCommand("self_attn.q_a_proj", rot, right_rotate=True))
    cmds.append(RotateCommand("self_attn.kv_a_proj_with_mqa", rot, right_rotate=True))
    cmds.append(RotateCommand("self_attn.o_proj", rot, right_rotate=False))
    if layer_index < first_k_dense_replace:
        cmds.append(RotateCommand("mlp.gate_proj", rot, right_rotate=True))
        cmds.append(RotateCommand("mlp.up_proj", rot, right_rotate=True))
        cmds.append(RotateCommand("mlp.down_proj", rot, right_rotate=False))
    else:
        cmds.append(RotateCommand("mlp.experts.gate_up_proj", rot, right_rotate=True))
        cmds.append(RotateCommand("mlp.experts.down_proj", rot, right_rotate=False))
        cmds.append(RotateCommand("mlp.shared_experts.gate_proj", rot, right_rotate=True))
        cmds.append(RotateCommand("mlp.shared_experts.up_proj", rot, right_rotate=True))
        cmds.append(RotateCommand("mlp.shared_experts.down_proj", rot, right_rotate=False))
        cmds.append(RotateCommand("mlp.gate", rot, right_rotate=True))
    # MLA 空间
    cmds.append(RotateCommand("self_attn.q_a_proj", rot_b_proj, right_rotate=False))
    cmds.append(RotateCommand("self_attn.q_b_proj", rot_b_proj, right_rotate=True))
    # kv_b 输出 v 部分旋转（[eye(qk_nope_head_dim), rot_uv] 块对角）
    cmds.append(RotateCommand(
        "self_attn.kv_b_proj",
        [torch.eye(qk_nope_head_dim, dtype=rot_uv.dtype, device=rot_uv.device), rot_uv],
        right_rotate=False,
    ))
    cmds.append(RotateCommand("self_attn.o_proj", rot_uv, right_rotate=True))
    # kv_a 输出 c 部分旋转（[rot_kv_b, eye(qk_rope_head_dim)] 块对角）
    cmds.append(RotateCommand(
        "self_attn.kv_a_proj_with_mqa",
        [rot_kv_b_proj, torch.eye(qk_rope_head_dim, dtype=rot_kv_b_proj.dtype, device=rot_kv_b_proj.device)],
        right_rotate=False,
    ))
    cmds.append(RotateCommand("self_attn.kv_b_proj", rot_kv_b_proj, right_rotate=True))
    return cmds


def get_glm52_ln_fuse_map(
    num_layers: int = 78,
    first_k_dense_replace: int = 3,
    num_experts: int = 256,
) -> Dict[str, List[str]]:
    """GLM-5.2 norm → 消费者线性映射（移植 get_ln_fuse_map，层内相对路径）。

    model.norm → lm_head 与 MTP norm 需在层外单独处理（见 fold_final_norm_lmhead）。
    """
    ln_map: Dict[str, List[str]] = {}
    for i in range(num_layers):
        ln_map[f"layers.{i}.input_layernorm"] = [
            f"layers.{i}.self_attn.q_a_proj",
            f"layers.{i}.self_attn.kv_a_proj_with_mqa",
        ]
        ln_map[f"layers.{i}.self_attn.q_a_layernorm"] = [f"layers.{i}.self_attn.q_b_proj"]
        ln_map[f"layers.{i}.self_attn.kv_a_layernorm"] = [f"layers.{i}.self_attn.kv_b_proj"]
        if i < first_k_dense_replace:
            ln_map[f"layers.{i}.post_attention_layernorm"] = [
                f"layers.{i}.mlp.gate_proj",
                f"layers.{i}.mlp.up_proj",
            ]
        else:
            ln_map[f"layers.{i}.post_attention_layernorm"] = [
                f"layers.{i}.mlp.experts.gate_up_proj",
                f"layers.{i}.mlp.shared_experts.gate_proj",
                f"layers.{i}.mlp.shared_experts.up_proj",
                f"layers.{i}.mlp.gate",
            ]
    return ln_map


# ---------------- 折叠执行 ----------------

def _get_child(layer: nn.Module, path: str) -> Optional[nn.Module]:
    mod = layer
    for part in path.split("."):
        mod = getattr(mod, part, None)
        if mod is None:
            return None
    return mod


def fold_glm52_layer(
    layer_module: nn.Module,
    layer_index: int,
    rotate_matrices: Dict[str, torch.Tensor],
    num_layers: int = 78,
    first_k_dense_replace: int = 3,
    num_experts: int = 256,
    qk_nope_head_dim: int = 192,
    qk_rope_head_dim: int = 64,
) -> List[Tuple[str, str]]:
    """对单个 GLM decoder 层做「norm 融合 + 旋转折叠」。

    返回 [(module_path, tensor_kind)]：被原地改写的模块清单
    （tensor_kind ∈ {weight, weight_3d}，供快照同步使用）。

    注意：模块名以「层内相对路径」给出，映射到 layer_module 子树。
    """
    ln_map = get_glm52_ln_fuse_map(num_layers, first_k_dense_replace, num_experts)
    touched: List[Tuple[str, str]] = []

    def _fuse_norm(norm_rel: str, targets: List[str]) -> None:
        norm_mod = _get_child(layer_module, norm_rel)
        if norm_mod is None or not hasattr(norm_mod, "weight"):
            return
        linears = []
        for t in targets:
            m = _get_child(layer_module, t)
            if m is None:
                continue
            if isinstance(m, nn.Parameter):
                # 3D 融合专家参数（gate_up_proj / down_proj 本体）
                if m.ndim == 3:
                    gamma = norm_mod.weight.data.float().to(m.device)
                    m.data.copy_((m.data.float() * gamma.view(1, 1, -1)).to(m.dtype))
                    touched.append((t, "weight_3d"))
            elif hasattr(m, "weight"):
                if m.weight.ndim == 2:
                    linears.append(m)
        if linears:
            fuse_ln_linear([norm_mod], linears)
            for t in targets:
                if _get_child(layer_module, t) is not None:
                    touched.append((t, "weight"))
            touched.append((norm_rel, "weight"))

    # norm 融合（quarot 全部融合，γ→1）
    for norm_rel, targets in ln_map.items():
        if norm_rel.startswith(f"layers.{layer_index}."):
            rel = norm_rel.split(".", 2)[2]  # 去掉 "layers.N." 前缀
            rel_targets = [t.split(".", 2)[2] for t in targets]
            _fuse_norm(rel, rel_targets)

    # 旋转折叠（本层命令，基于传入旋转矩阵）
    for cmd in _layer_commands(
        layer_index, rotate_matrices, num_layers, first_k_dense_replace,
        qk_nope_head_dim=qk_nope_head_dim, qk_rope_head_dim=qk_rope_head_dim,
    ):
        mod = _get_child(layer_module, cmd.target)
        if mod is None:
            continue
        if isinstance(mod, nn.Parameter):
            if mod.ndim == 3:
                rotate_weight_3d(mod, cmd.rot, right_rotate=cmd.right_rotate)
                touched.append((cmd.target, "weight_3d"))
            continue
        if not hasattr(mod, "weight"):
            continue
        if mod.weight.ndim == 2:
            rotate_linear(mod, cmd.rot, right_rotate=cmd.right_rotate)
            touched.append((cmd.target, "weight"))
        elif mod.weight.ndim == 3:
            rotate_weight_3d(mod.weight, cmd.rot, right_rotate=cmd.right_rotate)
            touched.append((cmd.target, "weight_3d"))

    return touched


def fold_embed_and_final(
    model: nn.Module,
    rotate_matrices: Dict[str, torch.Tensor],
    layer_name_prefix: str = "model",
) -> List[Tuple[str, str]]:
    """折叠残差流入口/出口（pre_run embed + post_run model.norm→lm_head）。

    - embed_tokens: right_rot=rot → 残差流进入旋转空间
    - model.norm → lm_head: norm γ 折进 lm_head 列、γ 置 1；lm_head right_rot=rot
      （残差流在旋转空间，lm_head 吸收）

    Args:
        model: 完整模型（或含 model.embed_tokens / model.norm / lm_head 的容器）
        rotate_matrices: get_glm52_rotate_maps 返回的旋转矩阵
        layer_name_prefix: 模型容器路径前缀（默认 "model"）
    """
    rot = rotate_matrices["rot"]
    touched: List[Tuple[str, str]] = []

    embed = _get_child(model, f"{layer_name_prefix}.embed_tokens")
    if embed is not None and hasattr(embed, "weight"):
        rotate_linear(embed, rot, right_rotate=True)
        touched.append((f"{layer_name_prefix}.embed_tokens", "weight"))

    norm = _get_child(model, f"{layer_name_prefix}.norm")
    lm_head = _get_child(model, "lm_head")
    if norm is not None and lm_head is not None and hasattr(lm_head, "weight"):
        fuse_ln_linear([norm], [lm_head])
        touched.append((f"{layer_name_prefix}.norm", "weight"))
        touched.append(("lm_head", "weight"))
        rotate_linear(lm_head, rot, right_rotate=True)
        touched.append(("lm_head", "weight"))

    return touched
