# Offload Trunk 设计文档

> 设计日期：2026-07-28
> 最后更新：2026-08-20
> 目标：在有限的昇腾 NPU HBM 资源上部署超出显存容量的量化大模型（如 GLM5.2 W8A8）
> 实现方式：基于 npuslim Patch 机制，无侵入增强 vllm-ascend 的权重 offload 能力

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
        NPUPrefetchOffloader（详见第 12 节：图模式 + TP>1 乱码问题）
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

### 10.2 不同策略示例

```bash
# size_aware（推荐）：自动计算 offload 量，交错布局
--additional-config '{"npuslim_offload_trunk": {"enabled": true}}'

# group：每 4 层 offload 1 层
--additional-config '{"npuslim_offload_trunk": {"enabled": true, "strategy": "group", "group_size": 4, "num_in_group": 1}}'

# custom：offload 所有层但保留 20-29 层
--additional-config '{"npuslim_offload_trunk": {"enabled": true, "strategy": "custom", "offload_layer_patterns": ["model.layers.[0-9]*"], "keep_layer_patterns": ["model.layers.2[0-9]"]}}'
```

### 10.3 测试脚本

```bash
# 启动服务（offload 50%，开启 trace）
NPUSLIM_OFFLOAD_TRACE=1 bash tests/plugins/test_offload_serve.sh 0.5

# 发送推理请求
bash tests/plugins/test_offload_infer.sh

```

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
| `offload/patch.py` | @register_patch 注册（offload_config 注入 + 异常处理） |

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

## 12. 已知问题与解决方案

### 12.1 图模式 + TP>1 下推理乱码

**现象**：使用 `--additional-config '{"npuslim_offload_trunk": {"enabled": true}}'` 启用 offload trunk，在图模式（非 `--enforce-eager`）且 `--tensor-parallel-size > 1` 时，推理输出为乱码（如重复的 `1`、`111111`、`…` 等）。TP=1 或 eager 模式下推理正常。

**根因**：

npuslim 的 `patched_init` 原实现在 `original_init` **之后**才调用 `set_offloader()` 设置 offloader。但 vllm 框架在 `gpu_model_runner.__init__` 内部会根据 `vllm_config.offload_config` 调用 `create_offloader()` 创建 offloader：

```
# vllm 框架内部 (gpu_model_runner.py):
set_offloader(create_offloader(self.offload_config))
# vllm-ascend (model_runner_v1.py):
if offload_cfg.prefetch.offload_group_size > 0:
    set_offloader(NPUPrefetchOffloader(...))
```

当用户通过 `--additional-config` 而非 `--offload_backend prefetch` 启用 offload 时，`offload_config.prefetch.offload_group_size` 为默认值 0，框架创建的是 `NoopOffloader` 而非真正的 prefetch offloader。

虽然 npuslim 后来用 `set_offloader()` 替换成了真正的 offloader，但 vllm 框架的其他部分已经基于 `offload_config` 做了决策（内存计算、图编译等），导致 offloader 与框架之间的不一致。在图模式 + TP>1 的场景下，这种不一致导致推理输出乱码。

**解决方案**：

在 `patched_init` 调用 `original_init` **之前**，根据 npuslim 的 OffloadPlan 设置 `vllm_config.offload_config` 的 prefetch 参数，让框架在 `__init__` 中就创建正确的 `NPUPrefetchOffloader`：

```python
def patched_init(self, vllm_config, *args, **kwargs):
    # 1. 解析配置、计算 budget、生成 plan
    # ...

    # 2. 设置 vllm 的 offload_config（必须在 original_init 之前）
    group_size = max(2, round(total_layers / num_offloaded))
    vllm_config.offload_config.offload_backend = "prefetch"
    vllm_config.offload_config.prefetch.offload_group_size = group_size
    vllm_config.offload_config.prefetch.offload_num_in_group = 1
    vllm_config.offload_config.prefetch.offload_prefetch_step = prefetch_step

    # 3. 调用原始 __init__ — 框架根据 offload_config 创建 NPUPrefetchOffloader
    original_init(self, vllm_config, *args, **kwargs)

    # 4. 用 EnhancedNPUPrefetchOffloader 替换原生 offloader
    #    （load_model 还没调用，替换是安全的）
    set_offloader(enhanced_offloader)
```

**验证**：Qwen3-30B-A3B，TP=2，`--gpu-memory-utilization 0.3`，offload 24/48 层，图模式，推理正常输出。

---

## 13. W4A8 量化模型（FRACTAL_NZ）Offload 支持 — K2.6 验证记录（2026-08-20）

### 13.1 背景与目标

trunk 此前已在非量化（Qwen3.8-27B bf16）和 W8A8（GLM5.2）模型上验证。Kimi K2.6 w4a8（W4A8 量化 MoE，61 层，499GB checkpoint，EP/DP4×TP8，4 节点 × 8×910B）暴露了一类新问题：**量化权重以 NPU internal format（FRACTAL_NZ）存储，使 trunk "plain ND 静态 buffer + plain `copy_`" 的核心假设失效**。

本轮目标：验证 offload trunk 的**量化 `wrapped_process_weights` 路径**在 K2.6 上端到端正确（功能正确性优先，性能后续再调）。

### 13.2 验证环境与配置

- 节点 10.42.0.70-73，容器 `vllm-ascend-zzw-v0.23.0`（910B 64GB）
- 部署改动：

| 文件 | 改动 |
|------|------|
| `vllm-learning/deploy_v2/models_config.sh` | `config_kimi2.6w4a8` 的 `GPU_MEM_UTIL` 0.90 → **0.50**（0.50 + `safety_margin_gb=14` 触发 trunk：权重 ~16GB/卡 offload ~2GB 后，KV 2.25GB/卡 通过 vllm 硬校验） |
| `vllm-learning/deploy_v2/kimi2.6w4a8/EP/start_ep.sh` | `NPUSLIM_PLUGIN_ENABLE=1`；`additional-config` 增加 `"npuslim_offload_trunk": {"enabled": true, "safety_margin_gb": 14}`；`cudagraph_mode` `FULL_DECODE_ONLY` → **NONE**（临时，见 13.4 坑 5）；移除 `DYNAMIC_EPLB=1`（vllm-ascend 对 K2.6 默认 False，保持 config 一致） |

- 规划结果（node0 日志，14:48 轮）：

```
OffloadPlan: offload=7/61 layers, prefetch_step=1, est_hbm=14.15GB, est_cpu=1.85GB,
             buffer=0.29GB, overlap=✓, source=auto, strategy=size_aware
Offloaded layer indices: [4, 13, 22, 30, 39, 48, 57]
FRACTAL_NZ pool slots for 7 unique param keys:
  mlp.experts.w13_weight, mlp.experts.w2_weight,
  mlp.shared_experts.down_proj.weight, mlp.shared_experts.gate_up_proj.weight,
  self_attn.fused_qkv_a_proj.weight, self_attn.o_proj.weight, self_attn.q_b_proj.weight
Initialized 7 modules. Total NPU memory saved: 2.2029 GB, Static buffer pool: 0.3101 GB
GPU KV cache size: 1,210,368 tokens
```

注意：每层 offload 的 **7 个参数全部是 FRACTAL_NZ**（MoE experts w13/w2 为 int32 `pack_to_int32` 输出，shared_experts 与 attention 为 int8）。

- 验证命令：`python vllm-learning/code/benchmark/quick_check.py --host 10.42.0.70 --port 9082 --model kimi_k26`

### 13.3 FRACTAL_NZ 关键语义（理解本问题的核心）

以下均为硬件实测/源码核实结论（探测脚本 `temp/probe_nz_*.py`、`temp/test_nz_capture.py`）：

1. **格式**：W4A8（`--quantization ascend`）权重经 `maybe_trans_nz`（vllm_ascend/utils.py:295）后是 **FRACTAL_NZ internal format**（`ACL_FORMAT_FRACTAL_NZ=29`）。`torch.npu.config.allow_internal_format=True`（vllm_ascend model_runner_v1.py:201 已设）是创建 NZ tensor 的前提。
2. **storageShape**：`torchair.core._npu_graph_executor.GetNpuStorageSizes(nz_tensor)` 对 NZ 返回 **5-D** 物理存储 shape；ND 返回 3-D/1-D。
3. **AIV op 硬要求**：`aclnnGroupedMatmulSwigluQuantWeightNzV2`（moe_mlp.py:273 `quant_apply_mlp` 调用）要求权重 storageShape 为 5-D（真 NZ）——plain ND 静态 buffer 直接报 **EZ1001 "storageShape must be 5, got [N], dimNum is 1"**。
4. **跨格式 `copy_`**：int8 层面 ND CPU → NZ NPU 的直接 `copy_` **数据正确**；**int32 层面**的跨格式 `copy_` **会损坏数据**（probe_nz_crosscopy.py 实测 C1 正确 / C3 错误）。
5. **view 保格式**：NZ int8 tensor 的 `.view(torch.int32)` 保留 FRACTAL_NZ 格式（仍是 5-D storage）——这正是 load 时 `pack_to_int32` 产出的 int32 权重也带 NZ 的原因；因此 int32 参数的 D2H/onload 必须落到 int8 base 上做。
6. **aclop 限制**：`npu_format_cast` 与跨格式 `copy_` 在 aclgraph（cudagraph）capture 中 dispatch "Identity" aclop 被拒绝（见坑 5）。
7. **格式观测时机**：D2H 只能发生在 `process_weights_after_loading` 期间（`wrapped_process_weights`）——这是 NZ 格式最后可被观测的时刻（搬走 NPU 侧张量后就没了）。

### 13.4 坑清单与解决（按发现顺序）

**坑 1：profile_run 崩溃 EZ1001 "storageShape must be 5"**
- 现象：启用 offload 后 profile_run 失败，`aclnnGroupedMatmulSwigluQuantWeightNzV2` 报权重 storage 非 5-D。
- 根因：`StaticBufferPool` 用 `empty_strided` 建 plain ND slot（1-D storage），不满足 WeightNz op 的 internal format 要求。
- 解决：`_NZStaticBufferPool` 对 NZ key 重建**真 NZ slot**：`npu_format_cast(torch.zeros(i8 base shape), 29)`；int32 权重在 int8 层面 cast 后 `.view(int32)`（view 保格式）。同时用 `i8_bases`（`data_ptr(slot) → int8 NZ base`）登记表供 onload 时定位 int8 base，避免运行时对 internal format tensor 做 `view()`。

**坑 2：int32 层面跨格式 copy 数据损坏**
- 实测：int32 ND CPU → int32 NZ NPU 直接 `copy_` 结果错误；改到 int8 层面正确。
- 解决：D2H 与 onload 的跨格式转换**全部在 int8 层面**进行。

**坑 3：`RuntimeError: expanded size (512) must match (2048)`（post_init）**
- 根因：`_nz_i8_handles` 快照在 `super().post_init()` **之后**才取，而第一次 onload（初始预取）在 `super().post_init()` **内部**就执行 → 找不到 int8 base，fallback 到错误 buffer。
- 解决：直接引用**活字典** `_NZStaticBufferPool.i8_bases`（池构造时填充，早于第一次 onload）+ lazy derive/cache 兜底。

**坑 4：`AttributeError: ... no attribute '_nz_scratch'`**
- 根因：中间版本 onload 用 3-step scratch 路径，清理时误删了 `__init__` 中的初始化。
- 解决：恢复初始化。最终版已简化为**单次 int8 层面跨格式 `copy_`**（坑 2 证明直接 copy 即可），不再需要 scratch，`_nz_scratch` 随之移除。

**坑 5：cudagraph capture 拒绝 "Cannot run aclop operators during NPU graph capture. aclop=Identity"**
- 现象：post_init + profile_run 全部通过后，cudagraph capture 阶段失败。
- 根因：NZ onload 序列（H2D copy / 跨格式 copy / `npu_format_cast`）dispatch "Identity" aclop，vllm 的 aclgraph capture 路径拒绝。
- 关键事实：**同一序列用独立 `torch.npu.NPUGraph` 可以捕获成功**（test_nz_capture.py）→ 拒绝与 vllm capture 上下文有关（stream/event/allocator 差异），精因待定。
- 规避（两层）：
  1. **专属 slot 设计**（代码层）：`nz_slot_count = offload 模块数`，`get_buffer` 按模块顺序**独占分配**（非循环复用）。slot 在 eager 阶段（post_init 初始预取 + profile_run）已填入真实权重 → **capture 与 replay 期间 NZ 参数零数据搬运**，`_patch_module_onload` 的 capture 分支只记录 event 协议（fork/copy_done 事件），不执行任何 NZ aclop。非 NZ 参数保持原生可捕获的 ND `copy_`（循环 slot，每次 replay 重填，与原生设计一致）。代价：buffer pool 从 1×max_layer 增至 7×max_layer（0.29→0.31GB，可忽略）。
  2. **临时 `cudagraph_mode=NONE`**（配置层）：先用 NONE 完成功能验证。
- 待办：恢复 `FULL_DECODE_ONLY` 验证专属 slot 设计下 capture 是否通过（理论上 NZ 路径已无 aclop）。

**坑 6（诊断方向）：误判为 EP dispatch 问题**
- 早期日志出现 "Split sizes" / "negative dimension" / FusedInfer 等错误，一度怀疑 EP dispatch 本身有问题。
- 基线实验（`GPU_MEM_UTIL=0.90` + trunk 关闭，13:04 轮）quick_check 通过 → 确认这些错误**全部由 offload 路径引入**（根因即坑 1/2 的 ND buffer 与 NZ 权重不匹配），EP dispatch 本身干净。
- 教训：offload 相关异常先做 trunk-off 基线对照，再下结论。

### 13.5 最终实现（代码级总结）

**`offload/patch.py` — `wrapped_process_weights`（D2H 阶段）**

对每个 offload 参数：
- NZ 检测：`"NZ" in str(torch_npu.get_npu_format(d))`，或名称兜底（`w13_weight`/`w2_weight` 且 dtype=int32）
- NZ 参数：记入 `offloader.nz_param_keys`，D2H 序列 =
  `d.view(int8)`（保留 NZ 5-D storage）→ `npu_format_cast(i8, ND)` → `.to("cpu")`（pinned N-D）→ `view(int32)`（若原为 int32）
- 非 NZ 参数：直接 `.to("cpu")`；大张量失败时走 `_d2h_chunked`（32MB 分块 N-D 拷贝兜底）

**`offload/npu_prefetch_offloader.py`**

- `_NZStaticBufferPool(StaticBufferPool)`：
  - `nz_names`（frozenset）+ `i8_bases`（data_ptr→int8 NZ base 活字典）
  - `__init__`：对 NZ key 将 slot 替换为 `_make_nz_slot`（int8 base → NZ cast → view int32；每模块专属 slot）
  - `get_buffer`：NZ key 按请求顺序独占分配（`_nz_assign_count`），非 NZ key 走原生循环分配
- `EnhancedNPUPrefetchOffloader.post_init`：
  - 若 `nz_param_keys` 非空：动态子类化 `_NZStaticBufferPool`，在 `super().post_init()` 期间 monkey-patch `vllm_ascend...prefetch.StaticBufferPool`（池类在该模块命名空间运行时查找），结束后还原并快照 `i8_bases`
- `_nz_onload(buf, cpu)`（eager）：单次 **int8 层面跨格式 `copy_`**：`i8_bases[buf.data_ptr()].copy_(cpu.view(int8), non_blocking=True)`
- `_patch_module_onload(mo)`：包装 `mo.start_onload_to_static`
  - 模块无 NZ 参数 → 走原生
  - capture 且 `mo._nz_slots_loaded` → **no-op 分支**：只记录 fork/copy_done event 协议，零数据搬运
  - 否则 → fork event + copy_stream 上逐参数（NZ 走 `_nz_onload`，非 NZ 走原生 `copy_`）+ done event
- `_log_buffer_diagnostics`：post_init 后打印每个大 buffer 的 shape / `GetNpuStorageSizes` / format（bufDiag）与 CPU storage 布局（cpuDiag），用于快速定位格式/布局问题

### 13.6 验证结果

- 部署成功：7/61 层 offload，NPU 节省 2.2029GB/卡，KV cache 1,210,368 tokens，服务 ready。
- quick_check **4/4 通过**（含任务中断后复查）：连贯的 Kimi 自我介绍 + reasoning，与 0.90+trunk-off 基线输出一致 → **offload 路径计算正确性验证通过**。
- 日志中仅有的 2 条 ERROR 均无害：usage 统计线程 cpuinfo JSON 解析失败（`_report_usage_worker` 后台噪音）；一次模型名拼错（`kimi-k26` vs `kimi_k26`）404。

### 13.7 性能（当前验证配置）

| 配置 | 130-150 token 请求 | TTFT |
|------|-------------------|------|
| offload + cudagraph NONE（本轮） | ~53-56s（≈0.5 tok/s） | 0.4-0.7s |
| 基线（0.90 + trunk off + FULL_DECODE cudagraph） | ~6.5s（≈22 tok/s） | 低 |

慢的构成：cudagraph 关闭 + 每个 decode step 都要 wait 7 层权重的 H2D（专属 slot 设计下 NZ 层每步重填 int8 copy，ND 层原生重填）。**功能正确性达标；性能需恢复 cudagraph 后再评估**——专属 slot 设计下 replay 期间 NZ 层零搬运，恢复 FULL_DECODE_ONLY 后 decode 应接近基线。

### 13.8 待办

> **2026-08-21 更新**：本节后 1/3 项已被第 14 节的图模式排查取代——capture 失败真因已定位（14.3），"专属 slot capture-safe"假设不成立（混合方案可行但净节省归零，14.6），图模式的完整方案见 14.7 决策矩阵。

1. `start_ep.sh` 恢复 `cudagraph_mode=FULL_DECODE_ONLY`，验证专属 slot NZ 设计的 capture + replay（理论已 capture-safe，实测确认）
2. 恢复 cudagraph 后复测性能，与基线对比
3. 若 capture 仍失败：对比 vllm aclgraph 与独立 `torch.npu.NPUGraph` 的 capture 上下文差异（stream/event/allocator）定位 Identity aclop 拒绝点
4. `GPU_MEM_UTIL=0.50` 为验证值（KV 仅 1.2M tokens），生产按需上调
5. 其他 w4a8 模型（GLM5.2 w4a8c8、Kimi K3）走同一量化路径，可直接复用本节 NZ 语义结论（13.3）与实现

### 13.9 改动文件清单

| 文件 | 改动 |
|------|------|
| `npuslim/src/npuslim/plugins/vllm_ascend/offload/npu_prefetch_offloader.py` | `_NZStaticBufferPool`（真 NZ slot + i8_bases + 专属分配）；`nz_param_keys`；`post_init` 池类替换；`_nz_onload`；`_patch_module_onload`（capture no-op 分支）；`_log_buffer_diagnostics` |
| `npuslim/src/npuslim/plugins/vllm_ascend/offload/patch.py` | `wrapped_process_weights` NZ 检测 + i8 层面 D2H；`_d2h_chunked` 分块兜底 |
| `vllm-learning/deploy_v2/models_config.sh` | K2.6 `GPU_MEM_UTIL=0.50` |
| `vllm-learning/deploy_v2/kimi2.6w4a8/EP/start_ep.sh` | offload 启用 + `safety_margin_gb=14` + 临时 `cudagraph_mode=NONE` + 移除 `DYNAMIC_EPLB=1` |
| `temp/probe_nz_*.py`（format/d2h/onload/crosscopy/final 等）、`temp/test_nz_*.py`（capture/rawbytes/roundtrip/probe 等） | NZ 语义硬件探测脚本（13.3 各结论的来源，关键：`probe_nz_crosscopy.py` 跨格式 copy 正确性、`test_nz_capture.py` 独立 NPUGraph 可捕获性） |

---

## 14. 图模式（cudagraph）× FRACTAL_NZ：`VLLM_ASCEND_ENABLE_NZ` 开关与 offload 形态（2026-08-21，Qwen3.6-27B-w8a8 三轮诊断）

### 14.1 背景

第 13 节用 `cudagraph_mode=NONE` 完成了 K2.6 的功能验证，遗留"图模式下 offload 能否工作"。本节回答该问题，并澄清一个关键疑问：**同样是量化模型，W8A8（qwen3.6 w8a8）之前"图模式没问题"，为什么 K2.6 W4A8 有问题？**

答案：之前的 W8A8 测试**没有启用 offload**。NZ 权重的 ND→NZ 转换只发生在 load 阶段（eager、capture 之前）一次；**无 offload 时 NZ 权重在图内是只读的**（图里只有消费 NZ 权重的 GEMM AIV op，本就为捕获设计），capture 天然无碍。问题只在 **NZ 权重 + offload（反复重载）+ 图模式** 三者同时出现时：重载路径需要 ND→NZ 转换，而该转换在 vllm 的图捕获上下文里不可捕获（14.3）。

### 14.2 `VLLM_ASCEND_ENABLE_NZ` 的语义（源码核实）

`_should_trans_nz`（vllm_ascend/utils.py:261）+ `weight_nz_mode`（ascend_config.py:249，env 默认 1，envs.py:92）：

| 权重 dtype | nz_mode=1（默认） | nz_mode=0 | nz_mode=2 |
|---|---|---|---|
| FP32 | 永不转 | 永不转 | 永不转 |
| BF16/FP16 | **不转** | 不转 | 转 |
| **量化 dtype（int8/int32，W8A8/W4A8）** | **默认转 NZ** | **不转** | 转 |
| 310P 机型 | 恒转 | — | — |

推论（全部实测验证）：

- **bf16 模型**（Qwen3.8-27B）：默认配置下权重全程 ND → offload = 原生"逐字节 DDR + 图内 `copy_`" → 与图模式天然兼容（0955 轮 FULL_DECODE_ONLY + offload 捕获成功 44s）。
- **量化模型**（W8A8/W4A8）：默认配置下 int8/int32 权重转 NZ → offload 重载引入不可捕获的转换 → 图模式撞墙（14.3）。
- **`VLLM_ASCEND_ENABLE_NZ=0`**：量化权重保持 ND，offload 退化为 bf16 模型的原生字节拷贝形态 → 图模式可用，且**全部 offload 体积都是真实节省**（R3 实测，14.6）。

注意：`VLLM_ASCEND_ENABLE_NZ=0` 使 W4A8 的 MoE GEMM 从 fused `grouped_matmul_swiglu_quant_v2`（WeightNz AIV op）落到 `quant_apply_mlp` 的 ND 回退分支（`npu_grouped_matmul`(scale+bias) + `npu_swiglu` + gmm2，moe_mlp.py）——功能正确（W8A8 已实测），性能损失待 benchmark（14.9）。

### 14.3 图模式 capture 失败的定位（69 节点空闲 NPU 隔离探针）

探针 `temp/probe_nz_vllmreplica.py` **逐行复刻 vllm 的真实捕获流程**（acl_graph.py + prefetch.py：默认流捕获（`torch.npu.graph(g, pool=...)` 不传 `stream=`）+ 共享 `graph_pool_handle` + 跨层 fork/wait/join 事件协议），逐一对照：

| 变体 | 结果 |
|---|---|
| c1: ND slot + 直接 `copy_`（原生 ND offload 路径） | **PASS**（Qwen3.8 同构） |
| c2: NZ slot + i8 直接跨格式 `copy_`（当前 `_nz_onload`） | **FAIL 107025**（PTA call acl api failed） |
| c3: NZ slot + 3-step（H2D + `npu_format_cast` + D2D copy） | **FAIL 107025** |

结论与修正（对 13.4 坑 5 的更新）：

1. **vllm 真实捕获上下文里，ND→NZ 转换（`npu_format_cast` / 跨格式 `copy_`）不可捕获**——K2.6 round 16 的 "Identity aclop" 真凶就是跨格式 copy 本身（`view()` 已被探针洗清）。
2. 同一序列在**独立 capture stream** 上可捕获（probe_nz_graphmode.py T1-T3）→ 行为与捕获流/pool 配置相关；PTA（aclgraph 引擎）内部规则是黑盒，只能经验规避。
3. 错误信息中的官方提示 `torch.npu.config.allow_internal_format=False` **不是解药**（probe_nz_flagfalse.py f1/f2）：capture 窗口置 False 救不了跨格式 copy_（仍 107025）；且该 flag 下 `npu_format_cast(zeros, NZ)` 直接产出 **ND 张量**（真 NZ 造不出来），slot 构造期必须保持 True。

**机制（为什么跨格式不可捕获，ND→ND 可以）**：纯 ND→ND H2D 是**内存搬运**，torch NPU 后端能翻译成 PTA 图原生支持的 DMA 拷贝节点（静态可表示）。ND→NZ 则要做 32×32 tile 布局变换，派发为无数学内容的"内部格式 acl op"（名为 **Identity**）——PTA 图构建器在 vllm 捕获上下文（默认流 + 共享 graph pool + fork/join 协议）下对这类 op **没有可录制的节点类型**：录制 → 无节点类型（107025）；当场执行 → capture 模式禁止现场执行 acl op（"Cannot run aclop..."）。两条路皆死。独立流可捕获证明非硬件不可能，而是 vllm 捕获配置的派发差异（根因在黑盒内）。绕开 torch 的 raw `aclrtMemcpy` 更危险：被 PTA **静默丢弃**、replay 不执行（14.4）——图内搬运必须走 torch 派发。

### 14.4 路径 B（物理 NZ 字节存 DDR + 无转换 raw DMA）——已证伪

用户直觉方案："load 已完成 NZ 转换，把 NZ 物理字节原样存 DDR，forward 时无转换写回"。数据层正确（同 5-D 几何，字节落回原位），但实现层被两堵墙挡住（probe_nz_flagfalse.py P1、probe_nz_rawmemcpy.py、probe_nz_rawreplay.py）：

| 步骤 | 结果 |
|---|---|
| P1: `nz.cpu()` | 驱动**解码为逻辑值**返回——torch API 拿不到物理字节 |
| r1/r2: ctypes `aclrtMemcpy`（`libascendcl.so`）raw D2H/H2D | eager 数据正确（写回真 NZ slot 后 `format_cast` decode == 原值） |
| rawreplay: 图内 raw H2D（vllm 复刻上下文） | **被 PTA 引擎静默丢弃**：capture 不报错，但 replay 时不执行（capture 时也未执行）——replay 后 slot 既非旧值也非新值。对照：torch `copy_` 是正常图节点（c1 PASS） |

结论：**不要**在图内用 ctypes/aclrt raw stream memcpy 搬数据（会被 PTA 丢弃且不报错，极难排查）；图内数据搬运只能走 torch 派发路径产生的图节点。

### 14.5 no-op 分支的两个 bug（R1 失败 EH0012）与修复

R1（NZ=1 + 第 13 节代码 + FULL_DECODE_ONLY）capture 0/5 失败，新错误签名（非 107025）：

```
Invalid_Argument(EH0012): aclrtAllocatorGetByStream failed.
  Reason: The stream is not registered with any allocator.
rtStreamWaitEvent execution failed, reason=in the model capture scenario,
  the event wait task has no corresponding event record task
```

两个 bug（`_patch_module_onload` 的 capture no-op 分支）：

1. **缺 fork**：no-op 分支只在 capture 期间对 `copy_stream` 做 `_copy_done_event.record(...)`，但没有先把 `copy_stream` fork 进捕获图（capture 流 `record_event` + `copy_stream.wait_event(fork)`）。结果 done-event 的 record 不是图节点 → PTA 收尾时 wait 无对应 record、side stream 无 allocator 注册 → EH0012。原生 full 路径有 fork，抄协议时漏了。
2. **ND 参数陈旧（正确性 bug，更隐蔽）**：混合模块（如 Qwen3.6 每层 = `linear_attn` bf16 ND + `mlp` int8 NZ）走 no-op 分支时，ND 参数的旋转 slot 不在图内回填 → replay 后所有层读 warmup 末层的陈旧权重，输出必错（即使 capture 不报错也是坏服务）。

**修复**（npu_prefetch_offloader.py）：no-op 分支 = fork（record + `copy_stream.wait_event`）+ **ND 参数图内 `copy_` 回填**（c1 已证可捕获，replay 正确）+ 仅 NZ 参数零搬运。

**残留风险**：纯 NZ 模块（如 K2.6 全部 offload 层）走 no-op 分支时 `copy_stream` 上无任何张量 op，EH0012 风险仍在（K2.6 当前 `cudagraph_mode=NONE` 不触发；K2.6 若要图模式应走 14.7 的路径 A）。

### 14.6 三轮诊断（Qwen3.6-27B-w8a8，69 节点 TP2，util 0.50 + margin 14，offload 41/64 层 9.6GB，FULL_DECODE_ONLY）

前提确认：W8A8 int8 权重（`mlp.gate_up_proj`/`down_proj`、`self_attn.qkv_proj`/`o_proj`）默认全部转 FRACTAL_NZ（preD2H 日志 `fmt=FRACTAL_NZ nz=True`）——**与 W4A8 同病**；之前 w8a8"没问题"只因没开 offload（14.1）。

| | R1（NZ=1 旧代码） | R2（NZ=1 no-op 修复后） | R3（NZ=0，路径 A） | R4（NZ=0 + util 0.62） |
|---|---|---|---|---|
| util / margin | 0.50 / 14 | 0.50 / 14 | 0.50 / 14 | **0.62** / 14 |
| FRACTAL_NZ | 确认（4 个 key） | 同 | **0 行**（全 ND） | 0 行 |
| offload 量 | 41/64 层 9.6GB | 同 | 同 | **6/64 层 1.2GB**（planner 自平衡，见 14.8.1） |
| capture | ❌ 0/5，EH0012（14.5） | ✅ 4s | ✅ 3s | ✅ 3s |
| 正确性 | — | ✅ 17×23=391、杭州 150 字连贯（`enable_thinking:false` 下 `finish:stop`） | ✅ 同 | ✅ 同 |
| 静态 pool | — | 日志 0.30GB（**父类 `total_bytes` 旧值**）/ 实测 **~7.5GB 专属常驻** | 0.30GB（真值，共享旋转） | 0.21GB |
| HBM 实测/卡 | — | 42GB | **34.8GB（少 7.3GB）** | 42.5GB（反升） |
| **净节省** | — | **~2GB**（仅 ND 部分；NZ 部分 offload 后被专属 slot 占回） | **~9.3GB（全部真实）** | ~1.2GB |
| KV cache | — | 18.96GiB / 525,697 tokens | 18.96GiB（相同，见 14.8.1） | 18.65GiB（未涨） |

内存账目（R2）：42GB ≈ 常驻权重 7.4 + KV 20.4 + 专属 NZ pool 7.5 + 激活/MTP/HCCL 等 ~7。R3 少掉的 7.3GB 与专属 pool 体量吻合——**混合方案下 offload 的 NZ 部分净节省归零**，只有 ND 部分（`linear_attn` bf16 等）真实省出。

### 14.7 图模式 offload 配置决策矩阵

| 形态 | capture | 净节省 | 性能 | 适用 |
|---|---|---|---|---|
| **A: `VLLM_ASCEND_ENABLE_NZ=0`（全 ND + 原生 offload）** | ✅（R3 实测） | **全部 offload 体积**（R3：9.3/9.6GB） | 量化 GEMM 走 ND 回退分支，损失待测 | **图模式 + offload 的正确形态**（W8A8 已验证；W4A8 待部署验证 14.9.2） |
| B: NZ=1 + 混合方案（14.5 修复后） | ✅（R2 实测） | 仅 ND 部分；**全 NZ 模块（K2.6）≈ 0** | 保留 NZ fused GEMM 快路径 | 需要 NZ 快路径性能、且模型 ND 占比可观、能接受大部分 offload 落空 |
| C: NZ=1 + `cudagraph_mode=NONE` | —（无图） | 全部（每步 eager 重载） | 最低（每步 H2D+转换无 overlap，K2.6 实测 ~0.5 tok/s） | 纯功能验证 / 不需要图的场景（K2.6 当前状态） |

### 14.8 当前约束与已知风险

1. **offload pool 不在 vllm 的 KV 账内 + planner 公式自平衡**：vllm 的 KV 上限 ≈ `total × util − (它追踪的常驻权重 + 激活峰值 + 固定开销)`，插件在记账之外分配的 pool 不可见（R2/R3 KV 同为 18.96GiB 的直接原因）。同时 npuslim planner 的公式（memory_budget.py）：
   ```
   available_for_weights = requested(=total×util) − kv_estimate − safety_margin
   required_offload      = total_weight − available_for_weights
   ```
   **抬 `GPU_MEM_UTIL` 不能增加 KV**（R4 实测证伪）：util 0.50→0.62 → required_offload 8.55→1.20GB（41→6 层）→ 常驻权重回升 → KV 上限不变（18.96→18.65GiB），HBM 实测反升（34.8→42.5GB）。
   - (a) R3 真实多省出的 ~7.2GB **不会自动变成 KV**——正确的杠杆是**抬 `safety_margin_gb`**（util 不变）：margin↑ → available_for_weights↓ → required_offload↑ → 常驻权重↓ → KV 上限↑（R5：util 0.50 + margin 21，预期 offload ~15GB / ~60 层 → KV ~26GB）；
   - (b) 形态 B（专属 pool ~7.5GB）下若 util 开高，vllm 会按"没有 pool"分 KV → **OOM 风险**。K2.6 B1/B2 两轮 OOM 的根因**不是** KV 而是插件专属 NZ slot 占回全部 offload 体积（14.10，已修复；曾误判为"KV 张量实际分配 ≈ 定尺 × 1.17"，B4 证伪：KV 张量精确按定尺分配）。**修复后的真实 KV 约束（B4 轮）**：free-memory 定尺 49.04 GiB 分配成功，但**首个 forward 的 PTA/EP 运行时工作集**（kernel workspace、EP alltoall 缓冲等，不在 profile 峰值内）吃掉剩余 ~6.4 GiB headroom → `ERR00100` OOM 引擎死。**结论：offload 后必须用 `--kv-cache-memory` 显式封顶，且在 权重+KV+profile 激活+non-torch 之外预留 ≥~7 GiB 运行时 headroom**（B3/B5：42 GiB 封顶，headroom ~13 GiB，服务验证正确）；
   - (c) 更干净的做法是 planner 直接接受"目标 KV 大小"输入（见 14.9.4），margin 是当前的间接杠杆。
   - (d) **margin 存在硬上限（planner 下限）**：模型有不可 offload 的最小常驻权重（Qwen3.6 实测 3.61GB：embedding/lm_head/MTP 等）+ pool，构成 `available_for_weights` 的下限（实测 ~3.82GB）。margin 抬到超出下限（R5：margin 21 → available 1.62GB < 3.82GB）时，planner 的 `validate_plan` 直接报 "Memory insufficient" **硬失败**（deficit 2.2GB），而不是把 required_offload 截断到可 offload 上限（16.96−3.82≈13.1GB）best-effort 降级——应改为 clamp + warning（见 14.9.4）。
2. **planner 的 buffer 估算失真**：plan 报 `buffer=0.21GB`（共享 slot 公式），形态 B 实际 ~7.5GB——预算决策应以 HBM 实测为准，planner 需按"NZ 专属 + ND 共享"修正（未改）。
3. **纯 NZ 模块的 no-op 分支有 EH0012 风险**（14.5 残留）：`copy_stream` 上无张量 op 时 side stream 可能不被 PTA 注册。形态 A 下无 NZ 参数，不触发。
4. **NZ=0 的 W4A8 ND 路径未实测**：`quant_apply_mlp` 的 ND 回退分支（`npu_grouped_matmul` scale+bias + `npu_swiglu` + gmm2）对 W8A8 已验证正确（R3），对 W4A8（int32 packed、per-channel scale_bias、`is_per_channel_weight`）未部署验证；gmm2 对 W4A8 ND 权重的接受性尤其待测。
5. **图内数据搬运必须走 torch 派发路径**：raw `aclrtMemcpy` 会被 PTA 静默丢弃（14.4.3）；跨格式转换 op 在 vllm 捕获上下文不可捕获（14.3）——两条边界内只有"ND → ND `copy_`"是图内合法的数据搬运。
   **推论（K2.6 w4a8 图模式 offload 不可行的完整链条）**：NZ 格式权重的每步 slot 回填（跨格式 `copy_`/3 步转换）不可捕获 → 图模式 + NZ = 不能真预取；于是图模式 + 全 NZ 模型只剩形态 B（专属 slot、图内零搬运），而专属 slot 把 offload 体积原样占回 HBM → 净节省 ≈ 0 或为负（B1/B2 OOM，14.10）；同时能产生净节省的形态 A（NZ=0 真 offload+真预取）被 W4A8 dispatch gap 阻断（14.8.4）。两条路皆死 → **图模式下 K2.6 不能 offload，要 offload 必须关图（形态 C，14.7）**。W8A8（Qwen3.6）无 dispatch gap，形态 A 两全（图 + 真 offload 已验证，15.2）。

### 14.9 遗留问题（按优先级）

1. ✅ **margin 杠杆已验证（R5b 完成）**：NZ=0 + util 0.50 + margin 18.5 → **offload 63/64 层 14.72GB（到地板附近）→ KV 23.72GiB / 658,178 tokens / 2.51x**（R3 margin 14：41 层 9.6GB / KV 18.96GiB / 2.01x；R4 util 0.62：证伪抬 util；R5 margin 21：触发 planner 下限硬失败，最小常驻权重 3.61GB）。63 层极限 offload 正确性抽检通过（17×23=391）。
2. ✅ **K2.6 显存压缩实验完成（15.1/15.3）**：基线 18.8 GiB/卡 → 最终配置（margin 52, 58/61 层, 形态 C + 42GiB KV 封顶）**2.16 GiB/卡，节省 16.64 GiB ≈ 88.5% ≈ 8.7 倍**（vllm 口径），quick_check 通过；期间发现并修复专属 NZ slot 无条件创建 bug（14.10），确认运行时 headroom 约束（14.8.1b）。最终配置服务已恢复并验证（quick_check ✓，集群留在 offload 状态）。
3. **插件估算偏差**（15.4 第 4 条，不影响对外数据）：`total_weight_per_card` 比 vllm 实测偏低 7–15%、`Total NPU memory saved` 多计；后续触碰插件时修正。
4. **ND vs NZ GEMM 性能 benchmark**：形态 A vs B（同模型同配置）量化性能差，作为生产选型依据。
5. **planner 修正**：(a) buffer 估算按"NZ 专属 slot（形态 B）/ 共享 slot（形态 A）"分别计算，形态 B 预算计入专属 pool；(b) required_offload 超可 offload 上限时 clamp + warning（而非硬失败，14.8.1d）；(c) 接口改为直接接受 `target_kv_gb`，margin 回归纯安全余量（14.8.1c）。
6. **纯 NZ 模块 no-op 分支加固**（仅当形态 B 要用于 K2.6 类全量化 MoE 时）：no-op 分支在 `copy_stream` 上放一个可捕获的最小 ND op 以注册 side stream，或改走形态 A。
7. **诊断配置恢复**：`qwen3.6_27b/TP/deploy_tp.sh`（`NPUSLIM_PLUGIN_ENABLE` / trunk / `VLLM_ASCEND_ENABLE_NZ=0` / margin 14）与 `models_config.sh`（util）为三轮诊断临时值。
8. **npuslim 提交**：第 13 节 5 文件改动 + 14.5 no-op 分支修复 + 14.10 专属 slot 修复 + 第 14/15 节文档（commit message 需补充 14.5/14.10/15 内容）。

### 14.10 专属 NZ slot 无条件创建 bug（2026-08-21 发现并修复）

**现象**：K2.6 形态 C（`cudagraph_mode=NONE` + trunk on）两轮 OOM（B1 margin 48 / B2 margin 52 + `--kv-cache-memory` 42GiB 封顶）。B2 账目：OOM 时 torch 已占 56.86 GiB = KV 42 + 常驻层 0.69 + **专属 NZ slot ≈ 18.25（= 插件 "Total NPU memory saved" 的全额）** + 插件未跟踪部分 ~2.8 + pool 0.31 + 激活 0.73——offload 的权重被 slot 原样占回，净节省为**负**，比不 offload 更差。

**根因**：`_NZStaticBufferPool`（npu_prefetch_offloader.py）按 `nz_slot_count = len(module_offloaders)` 为**每个 offload 模块**建独立真 NZ slot——该设计是形态 B（图捕获）专用的（捕获图内零数据搬运，14.3/14.5），但 `post_init` **不看 cudagraph_mode 无条件执行**。`cudagraph_mode=NONE` 时 eager 路径每步本来就要 H2D 重载（8/20 已验证 0.5 tok/s 即此行为），slot 纯属开销。这也解释了 14.7 矩阵"形态 B 对全 NZ 模型（K2.6）净节省 ≈ 0"——旧代码下**任何** cudagraph 模式的 K2.6 offload 净节省都 ≈ 0（或为负），形态 C 根本不存在。

**修复**（3 文件）：
- `config.py`：新增 `dedicated_nz_slots: Optional[bool]`（None=自动；additional-config `dedicated_nz_slots` 或 env `NPUSLIM_OFFLOAD_TRUNK_DEDICATED_NZ_SLOTS` 可强制覆盖）。
- `patch.py`：自动判定 = `cudagraph_mode` 字符串不含 `NONE`（跨版本稳健），日志打印决策。
- `npu_prefetch_offloader.py`：`EnhancedNPUPrefetchOffloader(dedicated_slots=...)`；`post_init` 按标志设 `nz_slot_count = len(module_offloaders) or 1`；关闭时 NZ key 走**单 slot 环形池**（eager 每步 int8 级跨格式 `copy_` 重载，8/20 已验证数据正确）；capture no-op 分支加 `and self.dedicated_slots` 守卫（强制 false + 开图时会 loud fail，绝不静默用脏数据）。

**影响面**：仅改变 `cudagraph_mode=NONE` 的 NZ 模型行为（形态 C 由此才真正存在）；形态 B（有图）行为不变。形态 C 性能 = 每步 H2D 重载（8/20 实测 ~0.5 tok/s @ 7 层；58 层会更慢，量级预期个位数 tok/s 以下——压缩与性能不可兼得，生产选型见 14.7）。

---

## 15. 权重压缩实验（HBM 压缩比）— 2026-08-21 完成

**目标**：在不改变模型规模与并行切分的前提下，offload trunk 能将**权重 HBM 占用**压缩多少（相对不 offload 的基线）。本章数据一律取 vllm 自身测量（日志 `worker.py:771` 行 `Actual usage: X GiB for weights`，每卡实际分配快照，基线/offload 两轮同一测量代码），npu-smi 只做旁证（reserved 口径含分配器持有块）。

### 15.1 最终压缩结果

| 模型 | 并行切分 | 采用形态 | 权重 HBM/卡：不 offload → offload 后 | 节省 | **压缩率** | offload 层数 | 正确性验证 |
|---|---|---|---|---|---|---|---|
| Qwen3.6-27B-w8a8 | TP2（node 69） | A：NZ=0 + 图 | 18.28 → **4.57 GiB** | 13.71 GiB | **75.0% ≈ 4.0 倍** | 63/64 | quick_check ✓ |
| Kimi K2.6 w4a8 | EP/DP4×TP8（node 70–73） | C：NZ=1 + 无图 | 18.8 → **2.16 GiB** | 16.64 GiB | **88.5% ≈ 8.7 倍** | 58/61 | quick_check ✓ |

- Qwen3.6：原来每卡占 18.28 GiB 权重，现在只需要 4.57 GiB（4 倍）；K2.6：原来 18.8 GiB，现在只需要 2.16 GiB（8.7 倍）。
- **两模型都到各自切分的压缩上限**：剩余常驻 = 不可 offload 地板（embedding/lm_head/MTP 等），margin 再加触发 planner 下限硬失败（14.8.1d）。
- K2.6 倍数更高：MoE 专家占比大、非层地板小；且同 util（0.90）下权重让出的 HBM 转为 KV（32.73 → 42 GiB，**+28% 并发容量**）。
- 压缩深度由 `safety_margin_gb` 控制（Qwen3.6：margin 14 → 41 层/9.6GB，18.5 → 63 层/13.7GB；K2.6：48 → 43 层，52 → 58 层到顶），抬 util 无效（R4 已证伪，14.8.1）。

### 15.2 当前实现情况（形态 × 模型）

| 形态 | 定义 | Qwen3.6（W8A8） | K2.6（W4A8） |
|---|---|---|---|
| **A**：NZ=0 + 图 + 真预取 | 图内 ND `copy_` 可捕获 → 图与压缩兼得，唯一两全形态 | ✅ **已验证采用**（75.0%/4.0 倍） | ❌ 被 vllm-ascend W4A8 dispatch gap 阻断（14.8.4）：上游 patch 解锁后仍需验证 W4A8 ND GEMM 路径 + benchmark |
| **B**：NZ=1 + 图 + 专属 slot | 图内零数据搬运，offload 体积被 slot 占回 | ⚠️ 仅 ND 部分有净节省 | ❌ 全 NZ → 净节省 ≈ 0（不采用） |
| **C**：NZ=1 + 无图 + 环形 slot | 每步 eager 跨格式重载，全量真实节省 | （不采用，A 更优） | ✅ **已验证采用**（88.5%/8.7 倍，当前 70–73 集群运行态） |

要点：
- 形态 C 的两个前提（均已落实）：插件 14.10 修复（`cudagraph_mode: NONE` 时不建专属 slot，`dedicated_nz_slots` 自动判定）；`--kv-cache-memory` 显式封顶并留 ≥ ~7 GiB 运行时 headroom（14.8.1b）。
- 代价与收益：形态 C 的 decode 无图 + 每步 H2D 重载（K2.6 实测 0.9 tok/s，基线 ~22 tok/s）；换来 16.64 GiB/卡权重 HBM + KV +28%。
- 生产选型：W8A8 类 → 形态 A（两全）；W4A8 全 NZ 类 → 容量优先用形态 C，性能优先等上游修 dispatch gap（14.8.4）后升形态 A。

### 15.3 复现配置（各模型最终配置）

环境：容器 `vllm-ascend-zzw-v0.23.0`（vllm-ascend v0.23.0，CANN 9.1.0，910B 64GB）；npuslim 共享盘 editable 安装（`/home/zzw/llm_infer_workspace/zzw/code/vllm-workspace/npuslim`），`NPUSLIM_PLUGIN_ENABLE=1` 时自动加载。部署脚本会被后续实验改写，复现以本节为准；全部轮次日志（含失败轮）归档于 `qwen3.6_27b/TP/log/` 与 `kimi2.6w4a8/EP/log/archive/`。

**Qwen3.6-27B-w8a8**（node 69，TP2）
公共：`--quantization ascend --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 8096 --enable-chunked-prefill --enable-prefix-caching --async-scheduling --speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`

| 参数 | 基线（不 offload） | offload 63/64（采用） |
|---|---|---|
| `--gpu-memory-utilization` | 0.90 | 0.50 |
| `VLLM_ASCEND_ENABLE_NZ` | 未设（=1） | **0** |
| additional-config | `{"enable_cpu_binding": true}` | + `"npuslim_offload_trunk": {"enabled": true, "safety_margin_gb": 18.5}` |

**Kimi K2.6 w4a8**（node 70–73，EP/DP4×TP8）
公共：`--max-model-len 256000 --max-num-seqs 32 --max-num-batched-tokens 8192 --data-parallel-size 4 --data-parallel-size-local 1 --data-parallel-rpc-port 13389 --tensor-parallel-size 8 --enable-expert-parallel --quantization ascend --block-size 128 --prefill-context-parallel-size 1 --decode-context-parallel-size 8 --cp-kv-cache-interleave-size 128 --enable-chunked-prefill --enable-prefix-caching --async-scheduling --enable-auto-tool-choice --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --mm-encoder-tp-mode data --seed 1024 --gpu-memory-utilization 0.90`
公共环境变量：`VLLM_ASCEND_ENABLE_MLAPO=1 HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=800 ASCEND_BUFFER_POOL=4:8 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TASK_QUEUE_ENABLE=1 ASCEND_AGGREGATE_ENABLE=1 ACL_OP_INIT_MODE=1 DYNAMIC_EPLB=1 OMP_NUM_THREADS=1`（`VLLM_ASCEND_ENABLE_NZ` 不设 = 1，W4A8 不能设 0，见 14.8.4）

| 参数 | 基线（不 offload） | offload 58/61（采用） |
|---|---|---|
| `cudagraph_mode` | FULL_DECODE_ONLY | **NONE** |
| additional-config | `{"enable_cpu_binding": true}` | + `"npuslim_offload_trunk": {"enabled": true, "safety_margin_gb": 52}` |
| `--kv-cache-memory` | 无 | **45097156608**（42 GiB 封顶，必须，14.8.1b） |

### 15.4 实验发现（结论已沉淀至 §14）

1. W4A8 + NZ=0 被 vllm-ascend dispatch gap 阻断（`moe_mlp.py:272` 不看权重格式恒选 NZ kernel），W8A8 无此问题 → 两模型形态选择差异的根源（14.8.4）。
2. 专属 NZ slot 只能在图捕获场景创建，无条件创建会让 offload 净节省为 0/负（14.10，已修复：`dedicated_nz_slots` 按 cudagraph_mode 自动判定）。
3. offload 后 KV 必须用 `--kv-cache-memory` 显式封顶，并为 PTA/EP 运行时工作集留 ≥ ~7 GiB headroom（14.8.1b；无封顶轮 KV 张量分配成功但首个 forward OOM）。
4. 对外数据一律用 vllm `for weights` 行：插件估算（`total_weight_per_card` 偏低 7–15%、`Total NPU memory saved` 多计）只用于 plan 决策；npu-smi 是 reserved 口径（含分配器持有块），只做旁证。
