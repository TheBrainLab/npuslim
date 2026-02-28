# QuIP: Quantization with Incoherence Processing

## 论文信息

- **标题**: QuIP: 2-Bit Quantization of Large Language Models With Guarantees
- **作者**: Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, Christopher De Sa (Cornell University)
- **链接**: https://arxiv.org/abs/2307.13304
- **代码**: https://github.com/Cornell-RelaxML/QuIP
- **发表**: 2023年7月

## 核心思想

QuIP 的核心洞察是：**量化受益于非相干(incoherent)的权重和 Hessian 矩阵**。

所谓"非相干"是指：
1. **权重均匀分布**: 权重幅度在各维度上分布均匀
2. **舍入方向无关**: 需要精确舍入的方向与坐标轴不对齐

## 算法流程

QuIP 包含两个主要步骤：

### Step 1: 自适应舍入 (Adaptive Rounding via LDLQ)

使用 LDLQ (Low-rank Decomposition Lattice Quantization) 算法进行 Hessian 感知的舍入：

```
目标: 最小化量化误差的二次代理目标
min ||W - W_q||_H^2 = (W - W_q) @ H @ (W - W_q).T
```

LDLQ 通过 Cholesky 分解 Hessian 矩阵，实现逐列的贪婪舍入，同时考虑已量化列的误差传播。

### Step 2: 非相干处理 (Incoherence Processing)

通过随机正交矩阵变换，确保权重和 Hessian 的非相干性：

1. **Rescale (对角缩放)**:
   ```python
   diagH = H.diag()  # Hessian 对角线
   diagW2 = (W.T @ W).diag()  # 权重二阶矩
   scaleWH = (diagH / diagW2).sqrt().sqrt()
   W = W * scaleWH[None, :]
   H = H / scaleWH[None, :] / scaleWH[:, None]
   ```

2. **Projection (随机正交投影)**:
   ```python
   # 使用 Butterfly 矩阵作为高效的正交变换
   U = random_butterfly_matrix(out_features)  # [m, m]
   V = random_butterfly_matrix(in_features)   # [d, d]
   W = U @ W @ V.T
   H = V @ H @ V.T
   ```

## 量化函数 (Quantization Function)

QuIP 支持两种量化函数：

### 1. MinMax 量化
```python
# Per-channel scale 和 zero
scale = (w_max - w_min) / maxq
zero = round(-w_min / scale)
q = clamp(round(w / scale) + zero, 0, maxq)
w_q = scale * (q - zero)
```

### 2. RMS 量化 (推荐，配合非相干处理)
```python
# 单一标量 scale
scale_rms = 2.4 * w.square().mean().sqrt()
w_normalized = w / scale_rms
q = clamp(((w_normalized + 1) / 2) * maxq, 0, maxq)
w_q = ((q / maxq) * 2 - 1) * scale_rms
```

**注意**: RMS 量化不需要 zero point，因为非相干处理已经让权重分布接近对称。

## 代码结构

```
quip/
├── __init__.py          # 模块导出
├── quip.py              # QuIP 主算法 (QuIPConfig, QuIP class)
├── quip_module.py       # QuIPModule (per-layer 量化逻辑)
├── vector_balance.py    # LDLQ 算法实现
└── README.md            # 本文档
```

### 关键类

1. **QuIPConfig**: 算法配置
   ```python
   @dataclass
   class QuIPConfig:
       w_bits: int = 4                    # 量化位数
       quant_func: str = "rms"            # "minmax" 或 "rms"
       ldlq_method: str = "ldlq"          # LDLQ 变体
       incoh_processing: bool = True      # 启用非相干处理
       preproc_rescale: bool = True       # 对角缩放
       preproc_proj: bool = True          # 随机投影
       fake_quant: bool = True            # 伪量化模式
   ```

2. **QuIPModule**: 单层量化
   - `fasterquant()`: 执行 LDLQ 量化
   - 调用 `preproc()` 和 `postproc()` 进行非相干处理

3. **LDLQConfig + quantize_weight_ldlq**: LDLQ 核心算法
   - 支持: ldlq, ldlqRG, allbal, ldl_gptqequiv 等变体

## 理论保证

QuIP 提供了首个 LLM 规模量化算法的理论分析：

1. **误差上界**: 量化误差与权重/Hessian 的相干性 (coherence) 相关
2. **非相干性**: 随机正交变换可以高概率降低相干性
3. **泛化性**: 理论同样适用于 GPTQ/OPTQ 等方法

## 当前实现状态

### 已完成
- [x] LDLQ 舍入算法 (vector_balance.py)
- [x] 非相干预处理 (base_hessian_module.py)
- [x] 伪量化模式 (fake_quant=True)
- [x] MinMax 和 RMS 量化函数

### 待完成 (真量化)
- [ ] QuIPLinear 层 (整数权重打包)
- [ ] 真量化模式 (fake_quant=False)
- [ ] 权重打包和解包逻辑
- [ ] 推理时反量化

## 真量化设计考量

### 核心挑战

伪量化流程：
```
W → preproc() → LDLQ → 反量化 → postproc() → W_q (float16)
```

真量化需要存储：
1. **整数权重** (qweight): LDLQ 舍入后的整数
2. **量化参数** (scales, zeros?): 反量化所需
3. **预处理参数** (projU, projV, scaleWH?): postproc 所需

**问题**: `projU` 和 `projV` 是 `[m×m]` 和 `[d×d]` 的正交矩阵，存储开销太大！

### 可能的解决方案

1. **方案A: 存储种子而非矩阵**
   - Butterfly 矩阵由随机种子生成
   - 推理时重新生成 projU, projV
   - 缺点：推理时需要额外计算

2. **方案B: 融合变换到权重**
   - 在量化后将 projU, projV 融合到权重
   - 推理时不需要额外存储
   - 需要仔细设计以保证数值稳定性

3. **方案C: 禁用 projection**
   - 只保留 rescale (scaleWH 很小)
   - 牺牲一些精度换取存储效率

## 参考文献

1. QuIP 原始论文: https://arxiv.org/abs/2307.13304
2. QuIP# (改进版): https://arxiv.org/abs/2402.04396
3. GPTQ: https://arxiv.org/abs/2210.17323
4. Butterfly 矩阵: https://arxiv.org/abs/1903.05885
