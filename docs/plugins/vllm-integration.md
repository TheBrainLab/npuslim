# vLLM 集成文档

## 1. 集成概览

NPUSlim 通过插件系统扩展 vLLM 的核心功能，主要包括三个方面：

- **MoE 模型补丁**：为 Qwen3-MoE、LongcatFlash 等模型提供自定义权重量化格式的加载支持。
- **推理执行器扩展**：修复 Ray 分布式执行器中的 rank 同步问题。
- **线性层权重加载扩展**：为堆叠（stacked）线性层提供自定义 merge 路径。

所有补丁通过 `vllm.general_plugins` 入口点自动加载，用户无需额外配置。

## 2. MoE 模型补丁

### 2.1 Qwen3-MoE 模型

**问题背景**：W4A16 量化为 MoE expert 权重添加了 `_packed`、`_scale`、`_shape`、`_offset` 后缀，上游 `Qwen3MoeModel.load_weights` 无法识别这些后缀参数。

**补丁位置**：`src/npuslim/plugins/vllm/model_executor/models/qwen3_moe.py`

**补丁逻辑**：

```python
@register_patch(
    target="vllm.model_executor.models.qwen3_moe",
    condition=package_version_range("vllm", max_version="0.20.1"),
)
def patch_qwen3_moe_load_weights(module):
    original_load_weights = module.Qwen3MoeModel.load_weights

    def patched_load_weights(self, weights):
        params_dict = dict(self.named_parameters())
        if _is_w4a16_quantized(params_dict):
            # W4A16 加载：尝试 _packed 后缀，再尝试 _scale/_shape/_offset
            return _w4a16_load_weights(self, weights, params_dict)
        return original_load_weights(self, weights)

    module.Qwen3MoeModel.load_weights = patched_load_weights
```

补丁检测到参数字典中包含 `experts.w13_weight_packed` 或 `experts.w2_weight_packed` 时，切换到 W4A16 专用加载路径。加载流程依次尝试以下后缀：

| 后缀类型 | 示例 |
|---------|------|
| 权重后缀 | `weight_packed`、`weight` |
| 辅助参数后缀 | `weight_scale`、`weight_shape`、`weight_offset` |

### 2.2 LongcatFlash 分组路由

**问题背景**：LongcatFlash 模型使用 Grouped Routing 机制（从 F 个分组中各选最优 expert 再 top-k），上游 vLLM 不支持该路由策略。

**补丁位置**：`src/npuslim/plugins/vllm/model_executor/models/longcat_flash.py`

**路由逻辑**：

```
Router 输出 (N+Z) 个 logits
  → reshape 为 F 个分组，每组 (N+Z)/F 个 expert
  → 每组选最高分 expert
  → 从所有组 winner 中选 top-k
  → 映射回原始 expert 索引 [0, N+Z)
```

**三种代码路径**：

| 场景 | 设备 | 处理方式 |
|------|------|---------|
| 无 zero expert | 任意 | 设置 `custom_routing_function`，路由工厂创建 `CustomRoutingRouter` |
| 有 zero expert | GPU | 直接 patch `ZeroExpertRouter._compute_routing`（路由工厂会忽略 `custom_routing_function`） |
| 任意 zero_expert_type | Ascend NPU | 替换 `AscendZeroExpertFusedMoE.select_experts`（其 `forward()` 绕过上述两条路径） |

补丁同时修改了 `FlashConfig.__init__` 以接受 `use_group_routing` 和 `expert_expansion_factor` 配置参数，以及 `LongcatMoe.__init__` 以根据配置注入分组路由。

### 2.3 线性层权重加载扩展

**问题背景**：Sparse24 稀疏量化需要将 tiled index 合并到 fused stacked buffer 中，上游 `QKVParallelLinear` 和 `MergedColumnParallelLinear` 的 `weight_loader` 不支持这种自定义合并路径。

**补丁位置**：`src/npuslim/plugins/vllm/model_executor/layers/linear.py`

```python
@register_patch(
    target="vllm.model_executor.layers.linear",
    condition=package_version_range("vllm", max_version="0.20.1"),
)
def patch_stacked_weight_loader_dispatch(module):
    """为 stacked linear loader 添加 opt-in 自定义 shard merge hook。"""
    # 检查参数是否携带 stacked_weight_loader 属性
    # 如果有，委托给自定义 loader；否则回退到原始实现
```

参数通过 `STACKED_WEIGHT_LOADER_ATTR` 属性声明自定义加载需求：

```python
# _stacked_sparse24.py 中定义
STACKED_WEIGHT_LOADER_ATTR = "stacked_weight_loader"

def load_stacked_sparse24_weight_index(layer, param, loaded_weight, shard_id):
    """将独立的 Sparse24 tiled index 合并到 fused stacked buffer。"""
    ...
```

## 3. 推理执行器扩展

### 3.1 Ray Worker Rank 同步

**问题背景**：`RayDistributedExecutor` 创建 worker 后按节点/IP 重排序，调用 `RayWorkerWrapper.adjust_rank()`。上游实现仅更新 `rpc_rank`，但 `global_rank` 保持旧值。当 PP > 1 时，worker 拥有不同层子集，`global_rank` 不一致会导致 KV-cache 配置索引错误。

**补丁位置**：`src/npuslim/plugins/vllm/executor/ray_utils.py`

```python
@register_patch(
    target="vllm.v1.executor.ray_utils",
    condition=package_version_range("vllm", max_version="0.20.1"),
)
def patch_ray_worker_adjust_rank(module):
    """同步 global_rank 与重排序后的 rpc_rank。"""
    original_adjust_rank = worker_cls.adjust_rank

    def patched_adjust_rank(self, rank_mapping):
        old_global_rank = self.global_rank
        original_adjust_rank(self, rank_mapping)
        if old_global_rank in rank_mapping:
            self.global_rank = rank_mapping[old_global_rank]

    worker_cls.adjust_rank = patched_adjust_rank
```

### 3.2 CGraph Trace（已禁用）

**位置**：`src/npuslim/plugins/vllm/executor/cgraph_trace.py`

此模块用于诊断 EP hang 问题，当前已通过 `always_disable` 条件禁用。EP hang 问题已定位为 Ray CGRAPH 与 vLLM-Ascend 集合通信交互导致，非 MoE 插件层问题。

## 4. 配置和使用方式

### 4.1 安装

```bash
# 安装 NPUSlim 及 vLLM 依赖
pip install npuslim[vllm]
```

安装后，vLLM 启动时会自动通过 entry_points 触发插件注册。

### 4.2 版本兼容性

| vLLM 版本 | 支持状态 | 说明 |
|-----------|---------|------|
| < 0.20.1 | 完全支持 | 所有 patch 处于活跃状态 |
| >= 0.20.1 | 部分支持 | patch 根据条件自动跳过，可能需要上游原生支持 |

### 4.3 日志排查

插件加载日志以 `[NPUSlimPatch]` 前缀输出，可通过以下方式确认插件状态：

```
[NPUSlimPatch] Discovered module: npuslim.plugins.vllm.model_executor.models.qwen3_moe
[NPUSlimPatch] Applied patch: patch_qwen3_moe_load_weights -> vllm.model_executor.models.qwen3_moe
[NPUSlimPatch] Applied 3 patch(es)
```

如需排查未生效的 patch，检查日志中的 "Skipped patch" 条目及跳过原因。

### 4.4 禁用特定补丁

在开发或调试场景下，可通过修改 `condition` 为 `always_disable` 来禁用特定补丁：

```python
from npuslim.plugins.registry import always_disable

@register_patch(target="some.module", condition=always_disable)
def patch_something(module):
    pass  # 不会执行
```
