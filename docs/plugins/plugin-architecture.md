# NPUSlim 插件架构总览

## 1. 设计理念

NPUSlim 采用 **运行时 Monkey-Patch** 策略扩展第三方框架（vLLM、Transformers、vLLM-Ascend 等），而非维护上游框架的 fork。

这种设计带来以下优势：

- **零侵入性**：不需要修改上游框架源码，上游版本升级时只需调整 patch 条件，无需重新 merge。
- **按需加载**：通过 entry_points 机制，仅在实际安装了目标框架时才加载对应插件，不引入多余依赖。
- **版本门控**：每个 patch 可声明生效的版本范围，上游 API 变更后自动跳过失效 patch，避免运行时崩溃。
- **可组合性**：多个 patch 独立注册，互不干扰，开发新 patch 不需要理解已有 patch 的内部逻辑。

## 2. 入口点机制

NPUSlim 通过 Python 的 `entry_points` 机制将自己的 `register()` 函数注册到目标框架的插件系统中。

### 2.1 pyproject.toml 定义

```toml
# vLLM 通用插件入口
[project.entry-points."vllm.general_plugins"]
npuslim = "npuslim.plugins:register"

# Transformers 量化器入口
[project.entry-points."transformers.quantizers"]
quip = "npuslim.plugins.transformers.quantizers.quantizer_quip:QuipHfQuantizer"
sparse24 = "npuslim.plugins.transformers.quantizers.quantizer_sparse24:Sparse24HfQuantizer"
```

### 2.2 调用链

以 vLLM 为例，完整的加载流程如下：

```
vLLM 启动
  └─ vLLM 加载 "vllm.general_plugins" entry_points
      └─ 调用 npuslim.plugins:register()
          ├─ 检测 vllm 可用 → 调用 npuslim.plugins.vllm.register()
          │   ├─ discover_modules() 扫描插件目录，触发 @register_patch 装饰器
          │   └─ apply_all_patches() 导入目标模块并执行 patch 函数
          ├─ 检测 npu 后端 + vllm_ascend 可用 → 调用 npuslim.plugins.vllm_ascend.register()
          │   ├─ discover_modules() 扫描 vllm_ascend 插件目录
          │   └─ apply_all_patches() 应用所有 vllm_ascend 补丁
          ├─ 检测 speculators 可用 → 调用 npuslim.plugins.speculators.register()
          └─ 始终调用 npuslim.plugins.transformers.register()
              └─ Transformers 量化器通过 entry_points 自动注册
```

### 2.3 register() 函数核心逻辑

```python
# src/npuslim/plugins/__init__.py
_REGISTERED = False

def register():
    global _REGISTERED
    if _REGISTERED:
        return  # 幂等：多次调用安全

    if _module_available("vllm"):
        _register_plugin("npuslim.plugins.vllm")

    _register_plugin("npuslim.plugins.transformers")

    if _load_backend_name() == "npu" and _module_available("vllm_ascend"):
        _register_plugin("npuslim.plugins.vllm_ascend")

    if _module_available("speculators"):
        _register_plugin("npuslim.plugins.speculators")

    _REGISTERED = True
```

关键设计点：

- **幂等性**：`_REGISTERED` 标志位确保 `register()` 多次调用不会重复执行。
- **条件加载**：通过 `_module_available()` 检测目标框架是否安装，按需加载。
- **后端感知**：vLLM-Ascend 插件仅在检测到 NPU 后端时加载。

## 3. register_patch 装饰器

`register_patch` 是插件系统的核心原语，用于声明对目标模块的补丁。

### 3.1 基本用法

```python
from npuslim.plugins.registry import register_patch

@register_patch(target="vllm.model_executor.models.qwen3_moe")
def patch_qwen3_moe_load_weights(module):
    """module 是已导入的目标模块对象，可直接修改其属性。"""
    original = module.Qwen3MoeModel.load_weights

    def patched_load_weights(self, weights):
        # 自定义逻辑
        return original(self, weights)

    module.Qwen3MoeModel.load_weights = patched_load_weights
```

### 3.2 工作原理

`register_patch` 有两种模式：

| 模式 | 参数 | 行为 |
|------|------|------|
| **延迟 patch** | `target` | 注册到全局 `_PATCH_REGISTRY`，由 `apply_all_patches()` 统一执行 |
| **即时注册** | `registrar` | 条件满足时立即调用 registrar（如 `@register_scheme`），在 import 时完成 |

```python
# 延迟 patch：target 模式
@register_patch(target="vllm_ascend.quantization.method_adapters")
def patch_method(module):
    module.AscendLinearMethod.create_weights = new_impl

# 即时注册：registrar 模式
@register_patch(
    registrar=register_scheme("W4A16", "linear"),
    condition=package_version_range("vllm_ascend", max_version="0.20.1"),
)
class AscendW4A16LinearMethod(AscendLinearScheme):
    ...
```

### 3.3 目标模块路径匹配规则

`target` 参数必须是 Python 模块的完整路径，使用 `.` 分隔：

```python
target="vllm.model_executor.models.qwen3_moe"       # 补丁 vllm 的 qwen3_moe 模块
target="vllm.v1.executor.ray_utils"                  # 补丁 vllm 的 ray_utils 模块
target="vllm_ascend.quantization.method_adapters"    # 补丁 vllm_ascend 的 method_adapters 模块
```

## 4. 版本门控机制

通过 `package_version_range` 条件函数控制 patch 的生效范围：

```python
from npuslim.plugins.registry import package_version_range, always_disable

# 仅在 vllm < 0.20.1 时生效
@register_patch(
    target="vllm.model_executor.models.qwen3_moe",
    condition=package_version_range("vllm", max_version="0.20.1"),
)
def patch_qwen3_moe(module):
    ...

# 始终禁用（用于调试或暂未启用的功能）
@register_patch(target="vllm.v1.executor.ray_utils", condition=always_disable)
def patch_cgraph_trace(module):
    pass
```

`package_version_range` 支持的参数：

- `min_version`：版本下界
- `max_version`：版本上界
- `include_min` / `include_max`：是否包含边界版本
- `version_attr`：读取版本号的属性名（默认 `__version__`）

当条件不满足时，patch 会被跳过并输出日志说明原因，不会抛出异常。

## 5. 文件路径约定

插件目录结构 **必须镜像** 目标框架的目录结构，以保持代码组织的一致性：

```
src/npuslim/plugins/
├── vllm/                                    # 补丁 vllm.* 模块
│   ├── executor/
│   │   ├── cgraph_trace.py                  # → vllm.v1.executor.ray_utils
│   │   └── ray_utils.py                     # → vllm.v1.executor.ray_utils
│   ├── model_executor/
│   │   ├── layers/
│   │   │   ├── linear.py                    # → vllm.model_executor.layers.linear
│   │   │   └── _stacked_sparse24.py         # 辅助模块（无对应 target）
│   │   └── models/
│   │       ├── qwen3_moe.py                 # → vllm.model_executor.models.qwen3_moe
│   │       └── longcat_flash.py             # → vllm.model_executor.models.longcat_flash
│   └── transformers_utils/
│       └── __init__.py                      # → vllm.transformers_utils
├── vllm_ascend/                             # 补丁 vllm_ascend.* 模块
│   ├── ops/
│   │   └── fused_moe/
│   │       └── zero_expert_fused_moe.py     # OOT 注册 ZeroExpertFusedMoE
│   └── quantization/
│       ├── method_adapters.py               # → vllm_ascend.quantization.method_adapters
│       └── methods/
│           ├── w4a16_linear.py              # register_scheme("W4A16", "linear")
│           └── sparse24_linear.py           # register_scheme("Sparse24", "linear")
├── transformers/                            # 补丁 transformers.* 模块
│   └── quantizers/
│       ├── quantizer_quip.py                # QuIP 量化器
│       └── quantizer_sparse24.py            # Sparse24 量化器
└── speculators/                             # 补丁 speculators 包
    └── __init__.py
```

## 6. 模块发现流程

`discover_modules()` 负责扫描插件目录并导入所有 Python 模块，从而触发模块中定义的 `@register_patch` 装饰器：

```python
def discover_modules(base_package: str, base_dir: str):
    """扫描 base_dir 下所有 .py 文件并导入。

    Args:
        base_package: 基础包名，如 "npuslim.plugins.vllm"
        base_dir: 基础目录路径
    """
    for py_file in base_path.rglob("*.py"):
        if py_file.stem == "__init__":
            continue
        module_name = f"{base_package}.{relative_path}"
        importlib.import_module(module_name)  # 触发 @register_patch
```

每个插件包的 `register()` 函数调用流程为：

```python
def register():
    discover_modules("npuslim.plugins.vllm", plugin_dir)  # 发现并注册所有 patch
    apply_all_patches()                                    # 统一应用
```

## 7. 日志系统

插件系统使用专用的 `patch_logger`，所有日志前缀为 `[NPUSlimPatch]`：

```python
from npuslim.plugins.logging import patch_logger

patch_logger.info("Discovered module: ...")
patch_logger.success("Applied patch: ... -> ...")
patch_logger.warning("Skipped patch: ... (condition returned False)")
```

## 8. 完整示例：创建新的 Patch

以下示例展示如何为 vLLM 添加一个新的模型补丁：

**第一步**：在镜像目录下创建补丁文件

```python
# src/npuslim/plugins/vllm/model_executor/models/my_model.py

from npuslim.plugins.logging import patch_logger
from npuslim.plugins.registry import package_version_range, register_patch


@register_patch(
    target="vllm.model_executor.models.my_model",
    condition=package_version_range("vllm", max_version="0.21.0"),
)
def patch_my_model_load_weights(module):
    """补丁 MyModel.load_weights 以支持自定义量化格式。"""

    original_load_weights = module.MyModel.load_weights

    def patched_load_weights(self, weights):
        params_dict = dict(self.named_parameters())
        # 检查是否使用目标量化格式
        if _is_custom_quantized(params_dict):
            return _custom_load_weights(self, weights, params_dict)
        return original_load_weights(self, weights)

    module.MyModel.load_weights = patched_load_weights
    patch_logger.info("Patched MyModel.load_weights for custom quantization support")
```

**第二步**：无需修改任何注册代码

由于 `discover_modules()` 会自动扫描 `model_executor/models/` 目录下的所有 `.py` 文件，新文件在插件加载时会被自动发现并注册。

**第三步**：验证

启动 vLLM 时观察日志输出：

```
[NPUSlimPatch] Discovered module: npuslim.plugins.vllm.model_executor.models.my_model
[NPUSlimPatch] Applied patch: patch_my_model_load_weights -> vllm.model_executor.models.my_model
```

## 9. 版本兼容策略

NPUSlim 插件系统采用以下策略处理上游版本演进：

| 场景 | 策略 |
|------|------|
| 上游 API 未变更 | patch 继续生效，无需修改 |
| 上游 API 已变更但 patch 仍可工作 | 放宽 `max_version` 条件 |
| 上游 API 已变更且 patch 不兼容 | 收紧 `max_version`，编写新 patch |
| 上游已原生支持相同功能 | 使用 `always_disable` 禁用 patch |

条件函数支持返回 `(bool, str)` 元组，提供跳过原因说明：

```python
def condition(module):
    if not hasattr(module.SomeClass, "target_method"):
        return False, "target_method not found in SomeClass"
    return True

@register_patch(target="some.module", condition=condition)
def patch(module):
    ...
```

这种策略确保 NPUSlim 在上游版本快速迭代时保持稳健，同时最小化维护成本。
