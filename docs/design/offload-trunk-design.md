# Offload Trunk 设计文档

> 设计日期：2026-07-28
> 最后更新：2026-08-22
> 目标：在有限的昇腾 NPU HBM 资源上部署超出显存容量的量化大模型（如 GLM5.2 W8A8）
> 实现方式：基于 npuslim Patch 机制，无侵入增强 vllm-ascend 的权重 offload 能力
>
> 文档分工：本文档只保留**设计方案**（1-11 章，含 2026-08-22 更新的部署约束矩阵 §10.2）。
> 调查过程（NZ/图模式/W4A8 根因、探针、实现过程）→ `offload-graphmode-investigation.md`；
> 测试结果（权重压缩比、性能、正确性、复现配置）→ `offload-trunk-test-results.md`。

---

## 1. 问题背景

### 1.1 场景

以 GLM5.2 为例：

| 资源 | 容量 |
|------|------|
| 计算节点 | 1 个 |
| NPU 卡数 | 8 |
| 每卡 HBM | 64 GB |
| 总 HBM | 512 GB |
| CPU DDR 可用 | ~1000 GB |
| GLM5.2 W8A8 权重 | ~720 GB |

权重 720 GB 超过总 HBM 512 GB，无法直接部署。

### 1.2 核心思路

将权重按层分为两部分：

- **常驻层（Resident）**：永久驻留在 HBM 中，无需传输
- **Offload 层**：存储在 CPU DDR 中，按需动态加载到 HBM

Offload 层共享一片 HBM 缓冲区（StaticBufferPool），通过异步预取（prefetch）与计算重叠，最小化性能损失。

### 1.3 与 vllm 现有机制的关系

vllm 已有完整的权重 offload 框架（`vllm/model_executor/offloader/`），vllm-ascend 已有 NPU 适配版本 `NPUPrefetchOffloader`。**npuslim 的角色是增强现有机制**——添加智能层选择、profiling 驱动优化和运行时监控，而非从零构建。

**内存管理边界**：offload trunk 只负责模型权重在 HBM 和 CPU 之间的分配。KV Cache、激活值等非权重内存由 vllm 框架根据 `--gpu-memory-utilization` 自动管理，offload trunk 不干预。

---

## 2. 底层机制：PrefetchOffloader 如何工作

要理解 npuslim offload trunk 的增强，首先需要理解 vllm/vllm-ascend 的 PrefetchOffloader 机制。

### 2.1 核心数据结构

```
┌─────────────────────────────────────────────────────────┐
│  NPU HBM (显存)                                          │
│                                                         │
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │  Resident 层权重  │  │ StaticBufferPool │              │
│  │  (永久驻留)       │  │ (offload 层共享)  │              │
│  │  layer 1, 3, 5.. │  │  slot 0          │              │
│  │                  │  │  (当前存放 layer 0│              │
│  │                  │  │   的权重副本)     │              │
│  └─────────────────┘  └─────────────────┘              │
│  ┌─────────────────┐                                    │
│  │  KV Cache        │  ← vllm 自动管理                   │
│  │  激活值           │                                    │
│  └─────────────────┘                                    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  CPU DDR (内存)                                          │
│                                                         │
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │  Offload 层权重   │  │  Offload 层权重   │              │
│  │  (pinned memory) │  │  (pinned memory) │              │
│  │  layer 0         │  │  layer 2         │              │
│  │  _cpu_storage    │  │  _cpu_storage    │              │
│  └─────────────────┘  └─────────────────┘              │
│  ┌─────────────────┐                                    │
│  │  Offload 层权重   │                                    │
│  │  layer 4         │                                    │
│  └─────────────────┘                                    │
└─────────────────────────────────────────────────────────┘
```

**三个关键概念**：

1. **Resident 层权重**：直接放在 HBM 中，forward 时直接读取，无需任何传输。例如上图的 layer 1, 3, 5...

2. **Offload 层的 `_cpu_storage`**：每层 offload 的权重存在 CPU pinned memory 中（pinned memory 可以实现异步 DMA 传输）。这是权重的"老家"。

3. **StaticBufferPool**：在 HBM 中预分配的一小块缓冲区，是 offload 层权重的"临时住所"。所有 offload 层**共享**这片缓冲区，通过循环复用（circular slots）来节省 HBM。`prefetch_step` 决定有多少个 slot——`prefetch_step=1` 表示只有 1 个 slot，所有 offload 层轮流使用同一片 HBM 空间。

### 2.2 权重加载阶段：如何分常驻和 Offload

整个过程发生在 `NPUModelRunner.load_model()` 中，按以下时序执行：

```
1. set_offloader(EnhancedNPUPrefetchOffloader)
   └─ 在 NPUModelRunner.__init__ 中完成
   └─ 此时 offloader 已持有 OffloadPlan（知道哪些层 offload）

2. get_model() → initialize_model() → make_layers()
   └─ make_layers() 内部调用 get_offloader().wrap_modules(layer_fn(...))
   └─ offloader 的 wrap_modules() 逐层处理:
      ├─ Resident 层: 直接创建在 NPU 上，param.data 在 HBM
      └─ Offload 层: 创建 _NPUModuleOffloader
         └─ _CpuParamOffloader.__init__():
            1. 在 CPU 上创建 pinned memory (_cpu_storage)
            2. 将 param.data 指向 _cpu_storage（而不是 HBM）
            3. 原始 HBM 上的空 tensor 被 GC 回收，释放 HBM

3. load_weights() → model.load_weights(weights)
   └─ 从 safetensors 逐 tensor 加载
   └─ default_weight_loader: param.data.copy_(loaded_weight)
      ├─ Resident 层: 权重直接写入 HBM 中的 param.data
      └─ Offload 层: 权重直接写入 CPU 中的 param.data（即 _cpu_storage）
         → 关键优势：offload 层的权重从磁盘直接到 CPU，不需要先到 NPU 再搬回！

4. process_weights_after_loading()
   └─ 量化后处理（repack/transpose 等），需要权重在 NPU 上
   └─ device_loading_context 逐模块处理:
      1. 将 offload 层的 param.data 从 CPU 临时搬到 NPU
      2. 执行 quant_method.process_weights_after_loading()
      3. 处理完后再搬回 CPU
      → 一次只需一层的 HBM 空间

5. post_init()  ← offloader 的核心初始化
   └─ ① sync_cpu_storage(): 同步量化后的权重到 _cpu_storage
   └─ ② 收集所有 offload 层的 ParamInfo (shape, dtype, stride)
   └─ ③ 在 HBM 上分配 StaticBufferPool:
      └─ 大小 = prefetch_step × max_offloaded_layer_size
      └─ 例：1 个 slot × 1.25 GB = 1.25 GB HBM
   └─ ④ 将每个 offload 层的 param.data 指向 HBM 中的 static buffer
      └─ 此时 param.data 在 HBM，但内容是未初始化的（需要 prefetch 填充）
   └─ ⑤ 启动初始预取: 把前 prefetch_step 个 offload 层的权重从 CPU 拷到 HBM buffer
```

### 2.3 推理阶段：计算与 Prefetch 如何交替

模型前向传播时，每层的 forward 被 offloader 的 hook 包裹：

```
model.forward(hidden_states):
  │
  ├─ Layer 0 (offloaded):
  │   ├─ torch.ops.vllm.wait_prefetch(hidden_states, idx=0)
  │   │   └─ 等待 copy_stream 上 layer 0 的 H2D 拷贝完成
  │   │   └─ 此时 HBM buffer slot 0 中已有 layer 0 的权重
  │   ├─ output = layer_0.forward(hidden_states)
  │   │   └─ 使用 HBM buffer 中的权重进行计算
  │   └─ torch.ops.vllm.start_prefetch(output, idx=1)
  │       └─ 在 copy_stream 上异步启动 layer 2 的 H2D 拷贝
  │       └─ copy_stream 和 compute_stream 并行执行！
  │
  ├─ Layer 1 (resident):
  │   └─ output = layer_1.forward(output)
  │       └─ 直接使用 HBM 中的权重，无需等待
  │       └─ 此计算期间，copy_stream 正在把 layer 2 权重从 CPU→HBM
  │       └─ 这就是"计算-预取重叠"窗口！
  │
  ├─ Layer 2 (offloaded):
  │   ├─ torch.ops.vllm.wait_prefetch(output, idx=1)
  │   │   └─ 等待 layer 2 的 H2D 拷贝完成
  │   │   └─ 如果 layer 1 的计算时间 ≥ H2D 拷贝时间，这里无需等待
  │   ├─ output = layer_2.forward(output)
  │   └─ torch.ops.vllm.start_prefetch(output, idx=2)
  │       └─ 异步启动 layer 4 的 H2D 拷贝
  │
  ├─ Layer 3 (resident):
  │   └─ ... (同 layer 1，提供下一个重叠窗口)
  │
  └─ ... 循环到最后一层，然后回到 layer 0（circular）
```

**两个 NPU Stream 的并行机制**：

```
compute_stream (默认):  [wait][compute L0][compute L1][wait][compute L2][compute L3]...
copy_stream (独立):           [copy L2 ← CPU..........][copy L4 ← CPU..........]
                               |←── 重叠 ──→|←── 重叠 ──→|
```

- `copy_stream` 是一个独立的 NPU Stream，H2D 拷贝在上面异步执行
- `compute_stream` 是默认的计算 Stream，矩阵乘法在上面执行
- 两个 Stream 通过 `Event` 同步：`wait_prefetch` 等 copy_stream 的 Event，`start_prefetch` 在 copy_stream 上记录 Event
- 当 resident 层在 compute_stream 上计算时，copy_stream 同时在拷贝下一个 offload 层的权重

### 2.4 Circular 缓冲区复用

所有 offload 层共享 `prefetch_step` 个 HBM buffer slot，循环复用：

```
prefetch_step=1 (1 个 slot):
  Layer 0 使用 slot 0 → 计算完后 → Layer 2 使用 slot 0 → ...
  HBM 开销: 1 × layer_size

prefetch_step=2 (2 个 slot):
  Layer 0 使用 slot 0, Layer 2 使用 slot 1 → Layer 4 使用 slot 0, Layer 6 使用 slot 1 → ...
  HBM 开销: 2 × layer_size (但可以预取更远，重叠更好)
```

### 2.5 vllm-ascend 的 NPU 适配

vllm 原生的 `PrefetchOffloader` 使用 `torch.cuda.Stream`/`torch.cuda.Event`。vllm-ascend 的 `NPUPrefetchOffloader` 将这些替换为 `torch.npu.Stream`/`torch.npu.Event`，其余逻辑完全一致。

`_NPUModuleOffloader` 是 vllm-ascend 中每个 offloaded 层的管理者，负责：
- 持有 `_cpu_storage`（CPU pinned memory 中的权重副本）
- 持有 `_gpu_buffer`（HBM static buffer 的引用）
- `start_onload_to_static()`：在 copy_stream 上执行 `gpu_buffer.copy_(cpu_storage, non_blocking=True)`
- 通过 `Event` 与 compute_stream 同步

---

## 3. npuslim 如何增强 PrefetchOffloader

### 3.1 原生 NPUPrefetchOffloader 的局限

| 局限 | 影响 |
|------|------|
| 只支持均匀分组（`module_index % group_size`） | MoE/Dense 混合层权重大小差异大，均匀分组效率低 |
| 无法按权重大小智能选择 | 可能 offload 了小层而保留了大层 |
| 无交错布局优化 | 连续 offload 多层时无计算-预取重叠 |
| 无 profiling 分析 | 不知道 H2D 拷贝是否能被计算隐藏 |
| 无运行时监控 | 无法看到 prefetch 效果 |

### 3.2 EnhancedNPUPrefetchOffloader 的增强

npuslim 的 `EnhancedNPUPrefetchOffloader` **不继承** `NPUPrefetchOffloader`，而是直接实现 `BaseOffloader` 接口，但**复用** vllm-ascend 的 `_NPUModuleOffloader`（每层的管理逻辑不变）。

增强点：

**1. 自定义层选择（替代均匀分组）**

原生：`module_index % group_size >= group_size - num_in_group`
增强：根据 `OffloadPlan.offload_layer_indices` 集合选择，可以是任意层组合

```python
# 原生 NPUPrefetchOffloader:
if module_index % group_size >= group_size - num_in_group:
    offload(module)

# EnhancedNPUPrefetchOffloader:
if module_index in self.plan.offload_layer_indices:  # 任意集合
    offload(module)
```

**2. 交错布局（size_aware 策略）**

通过 `OffloadPlanner._interleave_layers()` 将 offload 层均匀分布在模型中，保证相邻 offload 层之间有 resident 层提供计算-预取重叠窗口。

**3. Profiling 驱动优化**

模型加载后自动估算 H2D 时间和计算时间，如果发现重叠不足，自动减少 offload 层数。

**4. Trace 输出**

通过 `NPUSLIM_OFFLOAD_TRACE=1` 在推理过程中打印每层的 wait/prefetch 事件。

**5. 复用上游基础设施**

以下组件完全复用 vllm/vllm-ascend，不重新实现：
- `StaticBufferPool`（HBM 缓冲池分配）
- `_NPUModuleOffloader`（每层的 CPU 存储、H2D 拷贝、Event 同步）
- `_CpuParamOffloader`（pinned memory 管理、sync_cpu_storage）
- `torch.ops.vllm.wait_prefetch`/`start_prefetch`（自定义算子）
- `device_loading_context`（量化后处理的临时搬移）

### 3.3 Patch 注入机制

npuslim 通过 `@register_patch` 在 `NPUModelRunner.__init__` 和 `load_model` 中注入增强逻辑：

```
NPUModelRunner.__init__() 被 patch 后:
  1. 检查 npuslim offload trunk 配置
  2. 如果启用: 计算 MemoryBudget → 生成 OffloadPlan
  3. 设置 vllm_config.offload_config 的 prefetch 参数（group_size 等）
     ← 必须在 original_init 之前完成，让框架在 __init__ 中创建正确的
        NPUPrefetchOffloader（乱码问题的根因分析见
        offload-graphmode-investigation.md §6：图模式 + TP>1 乱码）
  4. 调用原始 __init__（框架根据 offload_config 创建 NPUPrefetchOffloader）
  5. set_offloader(EnhancedNPUPrefetchOffloader)  ← 替换为增强版
  → 后续 make_layers() 调用 get_offloader().wrap_modules() 时，用的是增强版

NPUModelRunner.load_model() 被 patch 后:
  1. 调用原始 load_model（加载权重、process_weights、post_init）
  2. 输出 OffloadMonitor 最终报告
```

---

## 4. 代码层次架构

```
src/npuslim/plugins/vllm_ascend/
├── offload/                             # offload trunk 功能模块
│   ├── __init__.py                      # 模块导出
│   ├── config.py                        # 配置 schema 与解析
│   ├── memory_budget.py                 # 内存预算计算（只管权重，KV Cache/激活值交给 vllm）
│   ├── planner.py                       # 智能 offload 规划器（三阶段决策 + 校验）
│   ├── npu_prefetch_offloader.py        # 增强的 NPU PrefetchOffloader（含 trace 输出）
│   ├── monitor.py                       # 运行时监控
│   └── patch.py                         # @register_patch 注册（offload_config 注入 + 异常处理）
```

各模块在整体流程中的位置：

```
patch.py
  ├─ patched_init() 调用:
  │   ├─ config.py          → 解析配置
  │   ├─ memory_budget.py   → 计算 HBM 和权重预算
  │   ├─ planner.py         → 生成 OffloadPlan（选哪些层）
  │   ├─ 设置 vllm_config.offload_config（必须在 original_init 之前）
  │   ├─ original_init()    → 框架根据 offload_config 创建 NPUPrefetchOffloader
  │   └─ npu_prefetch_offloader.py → 创建增强 offloader，set_offloader()
  │
  └─ patched_load_model() 调用:
      ├─ (原始 load_model: make_layers→wrap_modules→load_weights→post_init)
      └─ monitor.py         → 输出 OffloadMonitor 最终报告
```

---

## 5. 三种 Offload 策略详解

### 5.1 size_aware（推荐，默认策略）

**含义**：全自动计算 offload 量，按层权重大小选择哪些层 offload 到 CPU，并使用交错布局最大化计算-预取重叠。用户无需指定任何参数。

**工作原理**：

1. **精确估算 KV cache**：根据注意力类型（标准/MLA/SFA/DSA）和模型参数，计算 KV cache 占用的 HBM 大小

   ```
   page_size_bytes = block_size × num_kv_heads × head_dim × dtype_size
   blocks_per_layer = ceil(effective_max_model_len / (block_size × compress_ratio))
   total_kv_cache = num_attn_layers × blocks_per_layer × page_size_bytes
   ```

   其中 `effective_max_model_len = ceil(max_model_len / (DCP × PCP))`，不同注意力类型的 `head_dim` 计算方式不同（见第 7 节内存管理）

2. **计算权重可用空间**：

   ```
   requested_memory = total_hbm × gpu_memory_utilization
   available_for_weights = requested_memory - kv_cache - safety_margin
   ```

   `safety_margin`（默认 2GB）覆盖激活值和图编译开销的估算偏差

   **前置校验**：如果 `available_for_weights ≤ 0`，说明 HBM 连用户指定的 KV cache 都放不下，直接报错退出，提示用户减小 `max_model_len` / `max_num_seqs`、提高 `gpu_memory_utilization` 或增加卡数。必须先保证 KV cache 能完整存放，才继续计算权重 offload 量。

3. **自动计算 offload 量**：

   ```
   base_offload = max(0, total_weight_per_card - available_for_weights)
   if base_offload > 0:
       buffer_pool = prefetch_step × max_layer_size
       required_offload = max(0, total_weight_per_card + buffer_pool - available_for_weights)
   else:
       required_offload = 0
   ```

   其中 `buffer_pool` 按 `prefetch_step × 最大层大小` 预估（因为 planner 优先 offload 最大的层，buffer 必须容纳最大的 offloaded 层）。仅当权重确实放不下 HBM 时才计算 buffer pool 和 offload 量。

4. **CPU 内存检查**：如果 offload 量超过 CPU 可用内存的阈值（默认 60%），报错退出

5. **按大小选择 + 交错布局**：优先 offload 最大的层，然后均匀间隔分布

6. **prefetch_step 自适应**：如果 HBM 有余量，自动增大 prefetch_step（更多缓冲槽 = 更好的预取重叠）

**交错布局示例**（78 层模型，offload 5 层）：

```
连续布局（不好）: offload 0,1,2,3,4  → 层间无 resident，无重叠窗口
交错布局（好）:   offload 0,16,32,48,64  → 每 ~15 层 offload 1 层，中间 resident 提供重叠

执行时间线（交错）:
  wait(0)→compute(0)→start_prefetch(16)→compute(1~15,resident)→wait(16)→compute(16)→...
                                |←────── 重叠窗口 ──────→|
                                copy(16) 和 compute(1~15) 同时执行
```

**适用场景**：
- 大多数模型（homogeneous MoE 或混合 Dense/MoE）
- 用户不想手动指定 offload 参数
- 希望自动优化计算-预取重叠

**配置**：
```json
{"enabled": true}
```

最简用法，无需任何其他参数。程序自动估算 KV cache、计算权重可用空间、决定 offload 量。

### 5.2 group（均匀分组策略）

**含义**：将模型层按固定大小分组，每组 offload 最后一层。

**工作原理**：

- `group_size=N`：每 N 层为一组
- `num_in_group=M`：每组的最后 M 层被 offload

**示例**（48 层，group_size=4, num_in_group=1）：

```
Group 0: [0, 1, 2, ★3]    → offload layer 3
Group 1: [4, 5, 6, ★7]    → offload layer 7
Group 2: [8, 9,10, ★11]   → offload layer 11
...
offload: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47
```

**适用场景**：
- 需要精确控制 offload 的规律性布局
- 与 vllm 原生 PrefetchOffloader 兼容（行为一致）
- 模型各层大小均匀时的简单场景

**配置**：
```json
{"enabled": true, "strategy": "group", "group_size": 4, "num_in_group": 1}
```

### 5.3 custom（自定义层名策略）

**含义**：通过层名 pattern 精确指定哪些层 offload、哪些层保留。

**工作原理**：

- `offload_layer_patterns`：匹配的层会被 offload（支持 glob 和正则）
- `keep_layer_patterns`：匹配的层会被保留在 HBM 中（优先级高于 offload）

**示例**：

```json
{
  "strategy": "custom",
  "offload_layer_patterns": ["model.layers.[0-9]*"],
  "keep_layer_patterns": ["model.layers.2[0-9]"]
}
```

效果：offload 所有层，但保留 20-29 层在 HBM 中。

**适用场景**：
- 需要精确控制特定层的 offload 行为
- 实验不同 offload 布局对精度/性能的影响
- 混合架构模型中只想 offload MoE 层而保留 Dense 层

**配置**：
```json
{
  "enabled": true,
  "strategy": "custom",
  "offload_layer_patterns": ["model.layers.[0-9]*"],
  "keep_layer_patterns": ["model.layers.2[0-9]"]
}
```

### 5.4 策略对比

| 策略 | 层选择方式 | 重叠优化 | 用户复杂度 | 适用场景 |
|------|-----------|---------|-----------|---------|
| size_aware | 按大小 + 交错 | 自动 | 低（只需 offload_ratio） | 通用推荐 |
| group | 均匀分组 | 无 | 中（需 group_size） | 简单规律布局 |
| custom | 层名 pattern | 无 | 高（需写 pattern） | 精确控制 |

---

## 6. 两阶段决策机制

### 6.1 优先级体系

系统按两个层次决定"哪些层 offload 到 CPU"：

**第一层：用户指定配置（最高优先级）**

如果用户在配置中显式指定了 offload 方式，系统完全遵从。以下两种情况都属于用户显式配置：

- `strategy=custom`：用户用层名 pattern 精确指定了哪些层 offload
- `strategy=group`：用户用 group_size/num_in_group 指定了分组规则

**第二层：启发式自动规划（默认优先级）**

如果用户使用默认的 `strategy=size_aware`，系统根据模型 config 估算每层权重大小，自动计算需要 offload 多少层才能放下模型，然后用交错布局均匀分布 offload 层。这是最简单的用法——用户只需 `{"enabled": true}` 即可。

### 6.2 决策流程

```
NPUModelRunner.__init__
  │
  ├─ 解析 OffloadTrunkConfig
  ├─ 计算 MemoryBudget（从 NPU 硬件检测 HBM，估算权重大小）
  ├─ OffloadPlanner.plan() → 初始 OffloadPlan
  │   ├─ strategy=custom → 按用户 pattern 选择层
  │   ├─ strategy=group  → 均匀分组选择
  │   └─ strategy=size_aware → 按大小 + 交错布局
  ├─ validate_plan() → 内存校验
  │   ├─ 通过 → 继续
  │   └─ 不足 + strict → RuntimeError
  ├─ 设置 vllm_config.offload_config（group_size 等）
  ├─ original_init() → 框架创建 NPUPrefetchOffloader
  ├─ set_offloader(EnhancedNPUPrefetchOffloader)
  │
  └─ NPUModelRunner.load_model()
      ├─ get_model() → make_layers() → wrap_modules()
      │   → offloaded 层权重直接加载到 CPU pinned memory
      ├─ process_weights_after_loading() + device_loading_context
      ├─ post_init() → 分配 StaticBufferPool + 初始预取
      └─ 输出 OffloadMonitor 报告
```

---

## 7. 内存管理

### 7.1 职责划分

| 组件 | 管理内容 | 控制方式 |
|------|---------|---------|
| **offload trunk** | 模型权重在 HBM/CPU 之间的分配 | 自动计算（精确估算 KV cache 后取差值） |
| **offload trunk** | KV cache 需求估算 | 精确计算（标准/MLA/SFA/DSA 分别处理） |
| **vllm** | KV Cache 大小 | `--gpu-memory-utilization` 自动计算 |
| **vllm** | 激活值内存 | `profile_run` 自动测量 |
| **vllm** | 框架开销 | 自动管理 |

offload trunk 自动计算 offload 量，用户不需要指定 offload_ratio。计算公式：

```
requested_memory = total_hbm × gpu_memory_utilization
available_for_weights = requested_memory - estimated_kv_cache - safety_margin
base_offload = max(0, total_weight_per_card - available_for_weights)
if base_offload > 0:
    buffer_pool = prefetch_step × max_layer_size_per_card
    required_offload = max(0, total_weight_per_card + buffer_pool - available_for_weights)
else:
    required_offload = 0
```

其中 `estimated_kv_cache` 根据注意力类型精确计算：

| 注意力类型 | per-token KV cache | 适用模型 |
|-----------|-------------------|---------|
| 标准注意力 | `2 × (total_kv_heads // TP) × head_dim × dtype_bytes` | Qwen3, LLaMA |
| MLA | `(kv_lora_rank + qk_rope_head_dim) × dtype_bytes` | DeepSeek-V2/V3, GLM5 |
| SFA | `(kv_lora_rank + qk_rope_head_dim) × dtype + index_head_dim × dtype` | GLM5.2 |
| DSA | `(head_dim + index_head_dim) × dtype_bytes` | DeepSeek-V4 |

总 KV cache = `per_token × max_model_len × max_num_seqs`（考虑 DCP/PCP 分片）。

安全余量 `safety_margin`（默认 2GB）覆盖激活值和图编译开销——这些由 vllm 的 `gpu_memory_utilization` 自动管理，我们只留一个小余量防估算偏差。

### 7.2 CPU 内存检查

如果 offload 的权重超过 CPU 可用内存的阈值（默认 60%），系统会报错退出：

```
if required_offload > cpu_available × cpu_memory_threshold:
    raise RuntimeError("CPU 内存不足: 需要 X GB, 可用 Y GB, 阈值 Z%")
```

### 7.3 内存校验

在 plan 生成后，系统会校验：

```
total_hbm_needed = resident 层权重 + StaticBufferPool 缓冲池

if total_hbm_needed > available_for_weights:
    if strict_memory_check:
        raise RuntimeError("内存不足! 建议: 减少 max_model_len/max_num_seqs 或增加 NPU 卡")
    else:
        log.warning("内存不足!")
```

---

## 8. 配置设计

### 8.1 完整配置字段

通过 vllm 的 `--additional-config` 传入 JSON，或通过 `NPUSLIM_OFFLOAD_TRUNK_*` 环境变量设置。

```json
{
  "npuslim_offload_trunk": {
    "enabled": true,
    "strategy": "size_aware",
    "prefetch_step": 1,
    "safety_margin_gb": 2.0,
    "cpu_memory_threshold": 0.6,
    "strict_memory_check": true,
    "enable_monitor": true
  }
}
```

### 8.2 字段说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **核心配置** | | | |
| `enabled` | bool | false | 是否启用 offload trunk |
| `strategy` | str | "size_aware" | 策略：见第 5 节详解 |
| `prefetch_step` | int | 1 | 预取步数上限（实际自动适配 HBM 余量） |
| **内存** | | | |
| `safety_margin_gb` | float | 2.0 | 安全余量（覆盖激活值/图编译开销的估算偏差） |
| `cpu_memory_threshold` | float | 0.6 | CPU 内存使用阈值（超过则报错退出） |
| **group 策略专用** | | | |
| `group_size` | int | 0 | 每 N 层为一组 |
| `num_in_group` | int | 1 | 每组 offload 最后 M 层 |
| **custom 策略专用** | | | |
| `offload_layer_patterns` | list[str] | [] | offload 层名 pattern（glob/正则） |
| `keep_layer_patterns` | list[str] | [] | 保留层名 pattern（优先级高） |
| **参数级过滤** | | | |
| `offload_params` | set[str] | {} | 只 offload 匹配的参数（如 "w2_weight"） |
| **安全** | | | |
| `strict_memory_check` | bool | true | 内存不足时报错（false=只警告） |
| **监控** | | | |
| `enable_monitor` | bool | true | 启用运行时监控 |
| `monitor_log_interval` | int | 100 | 监控日志间隔（步数） |

---

## 9. 运行时 Trace

通过 `NPUSLIM_OFFLOAD_TRACE=1` 环境变量开启详细的 prefetch 动态换入换出日志。

**注意**：trace 开启时自动使用 `--enforce-eager` 模式（因为 loguru 的 `logger.info()` 会导致 `torch.compile` graph break）。关闭 trace 时使用 compile 模式，性能不受影响。

### 9.1 Trace 输出示例

```
# 模型加载阶段
[TRACE] wrap_modules: layer 0 (offloader_idx=0), 9 params, 1246.24 MB → CPU
[TRACE] wrap_modules: layer 2 (offloader_idx=1), 9 params, 1246.24 MB → CPU
[TRACE] post_init: initial prefetch → model.layers.0 (1246.24 MB CPU→NPU)

# 推理阶段（每个 forward step 循环遍历所有 offloaded 层）
[TRACE] step=0 📦 PREFETCH START → model.layers.2 (9 params, 1246.24 MB CPU→NPU)
[TRACE] step=0 ⏳ WAIT → model.layers.0 (self_attn, mlp, ...) | compute done | 🚀 PREFETCH → model.layers.2
[TRACE] step=0 📦 PREFETCH START → model.layers.4 (9 params, 1246.24 MB CPU→NPU)
[TRACE] step=0 ⏳ WAIT → model.layers.2 (...) | compute done | 🚀 PREFETCH → model.layers.4
```

---

## 10. 部署使用方式

### 10.1 基本使用

```bash
# 最简用法：自动计算 offload 量，无需指定任何参数
NPUSLIM_PLUGIN_ENABLE=1 vllm serve /path/to/model \
  --tensor-parallel-size 8 \
  --additional-config '{"npuslim_offload_trunk": {"enabled": true}}'

# 开启 trace（自动 eager 模式，调试用）
NPUSLIM_PLUGIN_ENABLE=1 NPUSLIM_OFFLOAD_TRACE=1 vllm serve /path/to/model \
  --tensor-parallel-size 8 \
  --additional-config '{"npuslim_offload_trunk": {"enabled": true}}'
```

### 10.2 约束关系（量化 × NZ × 图模式 × offload）

四个自变量：**量化类型**（非量化 bf16 / W8A8 int8 / W4A8 int4-packed-MoE）、**`VLLM_ASCEND_ENABLE_NZ`**（默认 1；控制量化权重是否转 FRACTAL_NZ 内部格式，bf16 权重永不转）、**图模式**（`cudagraph_mode`：`FULL_DECODE_ONLY` / `NONE`）、**offload trunk**（开/关）。

三条已验证的硬约束（推理与探针依据见 `offload-graphmode-investigation.md`）：

1. **图模式下 offload 的权重必须是 ND**：图内合法数据搬运只有 torch 派发的 ND→ND `copy_`；写入内部格式（FRACTAL_NZ）storage 的任意操作（跨格式 `copy_` / `npu_format_cast` / 纯 ND 原语分解）在 CANN 9.1.0 捕获下不可捕获（带 fork/join → 107025 `STREAM_UNJOINED`；未 fork → 静默丢弃不执行）。
2. **W4A8 per-channel MoE 在 910B 的 NZ=0 需要 nd_dispatch 补丁**：融合算子 `grouped_matmul_swiglu_quant_v2` 的 aclnn API 无条件强制 NZ 5-D storage（ND 权重 → EZ1001，profile_run 全 worker 崩溃）。trunk 随附 `offload/w4a8_nd_dispatch.py`：NZ=0 时自动把 per-channel W4A8 路由到 ND 回退分支（V5 `npu_grouped_matmul` A8W4 + AIV swiglu），kill switch `NPUSLIM_W4A8_ND_DISPATCH=0`。W8A8 与非量化模型不需要（W8A8 的 ND GEMM 原生支持）。
3. **offload 开启后 KV 必须显式封顶**：offload pool 不在 vllm 的 KV 记账内，必须 `--kv-cache-memory <bytes>` 封顶，并在 权重+KV+profile 激活+non-torch 之外留 ≥ ~7 GiB 运行时 headroom（首个 forward 的 torch_npu/EP 工作集不在 profile 峰值内）。

约束矩阵（910B 实测；✅ 已端到端验证，⚠️ 有条件可行，❌ 不可行）：

| 量化 | NZ | 图模式 | offload | 结论 | 形态 | 备注 |
|---|---|---|---|---|---|---|
| 非量化 bf16 | — | 任意 | 开 | ✅ | A'/C' | 权重全程 ND，原生形态（Qwen3.8 已验证） |
| W8A8 | **0** | 全图 | 开 | ✅ **采用** | **A** | Qwen3.6-27B：63/64 层，4.0 倍压缩，391 ✓ |
| W8A8 | 1 | 全图 | 开 | ⚠️ | B | 专属 slot：仅 ND 部分（bf16 attention 等）净节省；全 NZ 模块（K2.6 类）≈ 0 |
| W8A8 | 1 | 无图 | 开 | ✅ | C | 每步 eager 跨格式重载（K2.6 同款机制） |
| W4A8 MoE（K2.6） | **0** | 全图 | 开 | ✅ **采用** | **A** | 需 nd_dispatch 补丁 + **关 `DYNAMIC_EPLB`**（EPLB per-expert list 形态的 V5 scale 布局未定）；K2.6 实测压缩顶 **58/61 层，8.7 倍**（与无图同顶，391 ✓），margin 48 → 43 层/2.8 倍/1.3 tok/s（§10.2 注） |
| W4A8 MoE（K2.6） | 1 | 全图 | 开 | ❌ | — | 融合算子 NZ 强制（NZ=0 才可能绕开）+ 约束 1：图内 NZ 回填不可捕获 |
| W4A8 MoE（K2.6） | 1 | 无图 | 开 | ✅ **采用** | **C** | 58/61 层，8.7 倍压缩，~0.9 tok/s（当前生产选型：容量优先） |
| 任意 | 任意 | 任意 | 关 | ✅ | 基线 | 无 offload；量化模型图模式无额外限制（NZ 权重图内只读） |

**选型建议**：

- 非量化 bf16 → **形态 A**：NZ 设置无关（bf16 权重在 nz_mode=1/0 下都不转 NZ，§3.2 转换表；Qwen3.8 0955 轮默认配置 + 全图 + offload 验证）——默认配置即可。
- W8A8 → **形态 A**（`VLLM_ASCEND_ENABLE_NZ=0` + 全图 + offload）：图与压缩两全（int8 权重默认转 NZ，必须设 0）。
- W4A8（K2.6/K3/GLM5.2-w4a8 同类）→ **形态 A（补丁后）**（图 + 真 offload）或 **形态 C**（无图，NZ=1 原生路径）；两者压缩顶相同（g52 验证：同 margin 下均 58/61 层/8.7x，investigation §5.7），吞吐由 **H2D 搬运**决定（08-22 ISO2 实证：ND 未融合 GEMM ≈ 22-23 tok/s = NZ 基线，GEMM 格式无性能差）——margin 越大压缩越深、吞吐越低（43 层 1.3 / 58 层 0.9 tok/s），按容量-速度权衡选择 margin（investigation §5.6/§8.2）。
- 需要保留 NZ 内核路径（性能差异未验证的其他模型，或有其他 NZ 依赖）且要少量 offload（混合 ND/NZ 模型）→ 形态 B（专属 slot），注意其 HBM 账目（专属 pool 计入 offload 体积）。

### 10.3 各形态必配项速查

| 形态 | 必需配置 |
|---|---|
| **A（W8A8 / bf16）** | `NPUSLIM_PLUGIN_ENABLE=1`；`VLLM_ASCEND_ENABLE_NZ=0`；`cudagraph_mode=FULL_DECODE_ONLY`；`--kv-cache-memory <cap>`；`additional-config: {"npuslim_offload_trunk": {"enabled": true, "safety_margin_gb": <N>}}` |
| **A（W4A8）** | 上行全部 + 关闭 `DYNAMIC_EPLB`（不 export `DYNAMIC_EPLB=1`）；（nd_dispatch 补丁自动生效，`NPUSLIM_W4A8_ND_DISPATCH=0` 可整体禁用） |
| **B** | `NPUSLIM_PLUGIN_ENABLE=1`；NZ=1（默认）；`FULL_DECODE_ONLY`；`--kv-cache-memory <cap>`（专属 pool 账目）；`dedicated_nz_slots` 默认自动开 |
| **C** | `NPUSLIM_PLUGIN_ENABLE=1`；`cudagraph_mode=NONE`；`--kv-cache-memory <cap>`；NZ=1（W4A8 只能 NZ=1） |

补充：

- `safety_margin_gb` 是**压缩深度杠杆**（与可 offload 量 1:1）：margin↑ → offload 层数↑ → 权重 HBM↓，吞吐沿 H2D 带宽线下降（K2.6：48→43 层/2.8x/1.3 tok/s，52→58 层/8.7x/0.9 tok/s）；抬 `gpu_memory_utilization` 无效（R4 实测证伪，investigation §3.8）。**图模式不降低压缩顶**（g52 验证，investigation §5.7）。
- 压缩到顶的特征：剩余常驻 = 不可 offload 地板（embedding/lm_head/MTP 等）；margin 继续加大会触发 planner 下限硬失败（investigation §8 遗留项）。
- trace（`NPUSLIM_OFFLOAD_TRACE=1`）自动强制 eager（第 9 章），与图模式互斥。

### 10.4 不同策略示例

```bash
# size_aware（推荐）：自动计算 offload 量，交错布局
--additional-config '{"npuslim_offload_trunk": {"enabled": true}}'

# group：每 4 层 offload 1 层
--additional-config '{"npuslim_offload_trunk": {"enabled": true, "strategy": "group", "group_size": 4, "num_in_group": 1}}'

# custom：offload 所有层但保留 20-29 层
--additional-config '{"npuslim_offload_trunk": {"enabled": true, "strategy": "custom", "offload_layer_patterns": ["model.layers.[0-9]*"], "keep_layer_patterns": ["model.layers.2[0-9]"]}}'
```

### 10.5 测试脚本

```bash
# 启动服务（offload 50%，开启 trace）
NPUSLIM_OFFLOAD_TRACE=1 bash tests/plugins/test_offload_serve.sh 0.5

# 发送推理请求
bash tests/plugins/test_offload_infer.sh
```

各模型最终复现配置见 `offload-trunk-test-results.md` §5。

---

## 11. 关键文件索引

### npuslim offload trunk 模块

| 文件 | 职责 |
|------|------|
| `offload/config.py` | 配置 schema，从 additional_config 或环境变量解析 |
| `offload/memory_budget.py` | 内存预算计算（只管权重，KV Cache/激活值交给 vllm） |
| `offload/planner.py` | 智能 offload 规划器（三阶段决策 + 交错布局 + 内存校验） |
| `offload/npu_prefetch_offloader.py` | 增强的 NPU PrefetchOffloader（自定义层选择 + trace） |
| `offload/monitor.py` | 运行时监控 |
| `offload/patch.py` | @register_patch 注册（offload_config 注入 + 异常处理）；`wrapped_process_weights` NZ 检测 + i8 层面 D2H |
| `offload/w4a8_nd_dispatch.py` | W4A8 per-channel MoE 的 ND dispatch 补丁（NZ=0 时把融合算子调用路由到 V5 A8W4 ND 回退分支；形态 A 的 W4A8 必需件，见 §10.2 约束 2 与 investigation §5） |

### vllm / vllm-ascend 依赖

| 文件 | 作用 |
|------|------|
| `vllm/model_executor/offloader/base.py` | `BaseOffloader`, `set_offloader`, `get_offloader` |
| `vllm/model_executor/offloader/prefetch.py` | `PrefetchOffloader`, `StaticBufferPool` |
| `vllm/model_executor/offloader/prefetch_ops.py` | `wait_prefetch`, `start_prefetch` 自定义算子 |
| `vllm/model_executor/models/utils.py` | `make_layers()` — offloader 拦截点 |
| `vllm_ascend/model_executor/offloader/prefetch.py` | `NPUPrefetchOffloader`, `_NPUModuleOffloader` |
| `vllm_ascend/worker/model_runner_v1.py` | `NPUModelRunner` — patch 目标 |

### 测试脚本

| 文件 | 说明 |
|------|------|
| `tests/plugins/test_offload_serve.sh` | 端到端启动脚本 |
| `tests/plugins/test_offload_infer.sh` | 推理验证脚本 |

---


## 相关文档

| 文档 | 内容 |
|------|------|
| `offload-graphmode-investigation.md` | 调查过程记录：FRACTAL_NZ 语义与六坑（原 §13）、图模式×NZ 三轮诊断 + 捕获规则定案 + U/T 探针（原 §14）、W4A8 NZ 强制根因与 V5 A8W4 语义校准、路径 B（nd_dispatch）实现过程、图模式+TP>1 乱码（原 §12）、遗留问题与解锁路径 |
| `offload-trunk-test-results.md` | 测试结果：权重 HBM 压缩比（形态 A/B/C × Qwen3.6/K2.6）、性能吞吐、正确性验证、复现配置（原 §15 + 2026-08-22 新增轮次） |
