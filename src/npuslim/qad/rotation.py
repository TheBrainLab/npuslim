"""Hadamard 旋转矩阵生成（A7 消融 / 未来 CANN kernel 立项后的量化前端）。

移植自 msmodelslim `processor/quarot/common/quarot_utils.py` L36-112 与
`hadamard.py`（Mulan PSL v2，MindStudio 官方参考实现），2026-09-02 移植说明：

- 纯 torch 实现（Sylvester 蝶形），**不依赖 scipy**；参考实现的 txt 加载路径
  （非 2 幂尺寸）不移植——A7 只用 256/128 等 2 幂尺寸，非 2 幂直接报错
- `torch.manual_seed(seed)` 与参考 `seed_all` 对齐，同 seed 矩阵逐位一致
  （`random_hadamard_matrix` 只消费 torch 随机数）
- 约定：激活按行向量存（x @ W.T），旋转 online 形式为 x @ R.T；
  权重折叠为 W @ R.T（消费者侧），见 a7 harness

产物仅用于量化期消融与统计（§6.1），部署需在线 kernel（§2.5，非本文件范围）。
"""
from __future__ import annotations

import math
from enum import Enum

import torch


class QuaRotMode(Enum):
    HADAMARD = "hadamard"
    BLOCK_HADAMARD_SHIFTED = "block_hadamard_shifted"


def is_pow2(n: int) -> bool:
    return (n & (n - 1)) == 0 and n > 0


def _matmul_had_u(x: torch.Tensor, size: int) -> torch.Tensor:
    """蝶形 Walsh-Hadamard 变换（行向量约定），末维归一化 1/sqrt(n)。

    移植自参考实现 matmul_had_u 的 k=1 分支（2 幂尺寸无需 txt 基矩阵）。
    """
    inp = x.clone().view(-1, size, 1)
    out = inp.clone()
    while inp.shape[1] > 1:
        inp = inp.view(inp.shape[0], inp.shape[1] // 2, 2, inp.shape[2])
        out = out.view(inp.shape)
        out[:, :, 0, :] = inp[:, :, 0, :] + inp[:, :, 1, :]
        out[:, :, 1, :] = inp[:, :, 0, :] - inp[:, :, 1, :]
        out = out.view(inp.shape[0], inp.shape[1], -1)
        inp, out = out, inp
    return inp.view(x.shape) / math.sqrt(size)


def random_hadamard_matrix(size: int, dtype: torch.dtype = torch.float32,
                           device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """随机符号对角 × 归一化 Hadamard：R = H_walsh · diag(±1) / sqrt(n)。

    注意：调用方负责先 torch.manual_seed（与参考 create_rot 内 seed_all 对齐）。
    """
    with torch.device(device=device):
        rot = torch.randint(low=0, high=2, size=(size,)).to(dtype)
        rot = rot * 2 - 1
        rot = torch.diag(rot)
        return _matmul_had_u(rot, size).to(device)


def random_hadamard_matrix_block(in_channel: int, stride: int, eye_step: tuple,
                                 dtype: torch.dtype = torch.float32,
                                 device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """对角块 Hadamard（部分块可置单位阵），移植自参考实现同名函数。"""
    h_stride = random_hadamard_matrix(stride, dtype, device)
    h_full = torch.zeros((in_channel, in_channel), device=device, dtype=dtype)
    for i in range(in_channel // stride):
        s, e = i * stride, (i + 1) * stride
        h_full[s:e, s:e] = torch.eye(stride, device=device, dtype=dtype) if i in eye_step else h_stride
    return h_full


def create_rot(mode: QuaRotMode, size: int, block_size: int = -1, rot_step: int = 1,
               eye_step: tuple = (-1,), seed: int = 1234,
               dtype: torch.dtype = torch.float32,
               device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """创建旋转矩阵。语义与参考实现逐行对齐（含 shift=16、rot_step 1-3）。

    - HADAMARD：block_size=-1 全维 Hadamard；否则 block_diag(size//block_size 块)
    - BLOCK_HADAMARD_SHIFTED：对角块 + shift 循环置换交织（rot_step 层数）
    """
    torch.manual_seed(seed)  # 对齐参考 seed_all 中实际被消费的 torch 种子

    if mode == QuaRotMode.HADAMARD:
        dim = size if block_size == -1 else block_size
        rot = random_hadamard_matrix(dim, dtype, device)
        if block_size != -1:
            rot = rot.repeat(size // block_size, 1, 1)
            rot = torch.block_diag(*rot)
        return rot

    if mode == QuaRotMode.BLOCK_HADAMARD_SHIFTED:
        block_size = 32 if block_size == -1 else block_size
        rot = random_hadamard_matrix_block(size, block_size, eye_step, dtype, device)
        identity = torch.eye(size, dtype=torch.float32)
        shift = 16
        p = torch.cat((identity[:, -shift:], identity[:, :-shift]), dim=1).to(rot.device)
        if rot_step == 1:
            return rot @ p @ rot
        if rot_step == 2:
            return rot @ p @ rot @ p.T @ rot.T
        if rot_step == 3:
            return rot @ p @ rot @ p.T @ rot.T @ p @ rot @ p.T @ rot.T
        raise ValueError(f"rot_step must be 1, 2, or 3, got {rot_step}")

    raise ValueError(f"unsupported QuaRotMode: {mode}")
