# 自定义算子（Custom Ops）内部机制

## 概述

`ops` 模块位于 `src/npuslim/ops/`，包含 NPUSlim 针对特定硬件平台加速的自定义算子实现。当前主要提供面向昇腾 NPU 的 4:2 结构化稀疏矩阵乘法算子 `sparse_matmul`。

模块的设计原则是**懒加载与优雅降级**：如果底层 `.so` 未编译或不可用，模块仍然可以正常导入，但调用算子时会抛出 `RuntimeError`，不会导致导入期崩溃。

## 模块组织结构

```
src/npuslim/ops/
├── __init__.py                  # 模块说明文档字符串
├── _platform.py                 # 共享平台检测工具
└── sparse_matmul/
    ├── __init__.py              # Python 绑定 + torch.library 注册
    ├── build.py                 # 编译构建脚本
    └── csrc/
        ├── CMakeLists.txt       # CMake 构建配置
        ├── sparse_matmul_host.cpp           # Host 侧入口
        ├── matmul_sparse_custom_kernel.h    # AscendC Kernel 声明
        ├── matmul_sparse_custom_kernel.cpp  # AscendC Kernel 实现
        ├── matmul_sparse_custom_tiling.h    # Tiling 数据结构
        └── matmul_sparse_custom_tiling.cpp  # Tiling 生成逻辑
```

## 平台检测：_platform.py

`_platform.py` 提供硬件环境检测函数，仅依赖标准库，不触发 `npuslim.__init__` 的导入。

### is_npu_available

```python
def is_npu_available():
    """检查 NPU/CANN 环境是否存在。"""
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    return bool(ascend_home) and os.path.isdir(ascend_home)
```

### is_gpu_available

```python
def is_gpu_available():
    """检查 CUDA 环境是否存在。"""
    cuda_home = os.environ.get("CUDA_HOME", "") or os.environ.get("CUDA_PATH", "")
    if cuda_home and os.path.isdir(cuda_home):
        return True
    # 回退：尝试运行 nvidia-smi
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True, timeout=5)
        return True
    except Exception:
        return False
```

### detect_soc_version

```python
def detect_soc_version():
    """通过 acl.get_soc_name() 检测昇腾 SoC 版本。"""
    result = subprocess.run(
        [sys.executable, "-c", "import acl; print(acl.get_soc_name())"],
        capture_output=True, text=True, timeout=10,
    )
    # 返回 e.g. "Ascend910_9392" 或 None
```

## sparse_matmul 算子

### 概述

`sparse_matmul` 实现了面向昇腾 NPU 的 **4:2 结构化稀疏矩阵乘法**。该算子利用 AscendC 编程模型直接驱动 Cube 单元执行稀疏矩阵乘法，绕过稠密计算的冗余计算量。

### 算子接口

```python
from npuslim.ops.sparse_matmul import sparse_matmul_4to2

# a:       [M, K] int8 NPU tensor
# b_dense: [N, K//2] int8 NPU tensor —— 4:2 稀疏 B 的稠密表示
# index:   NZ-fractal 格式索引张量 (uint8)，位于 NPU
c = sparse_matmul_4to2(a, b_dense, index)  # -> [M, N] int32 NPU tensor
```

### 约束条件

- `a` 和 `b_dense` 必须为 2D 张量
- `K % 4 == 0`（4:2 稀疏要求）
- `b_dense.shape == [N, K // 2]`
- 所有张量必须位于 NPU 设备上

### 运行时加载机制

```python
_LIB = None
_LIB_LOADED = False

def _load_lib():
    """尝试加载 libsparse_matmul.so，幂等操作。"""
    global _LIB, _LIB_LOADED
    if _LIB_LOADED:
        return
    _LIB_LOADED = True
    so_path = os.path.join(os.path.dirname(__file__), "libsparse_matmul.so")
    if not os.path.isfile(so_path):
        return  # .so 不存在，优雅降级
    _LIB = ctypes.CDLL(so_path)
```

### torch.library 注册

如果 `.so` 加载成功，模块会通过 `torch.library.custom_op` 注册自定义算子，并注册 fake kernel 用于 `torch.compile` / Dynamo 追踪兼容。

### 构建系统

```bash
python -m npuslim.ops.sparse_matmul.build
```

构建流程：

1. **平台检查**：验证 `ASCEND_HOME_PATH` 和 SoC 版本
2. **CMake 配置**：传递 `RUN_MODE=npu`、`SOC_VERSION`、`ASCEND_CANN_PACKAGE_PATH`
3. **编译**：AscendC Kernel 编译为静态库，Host Wrapper 编译为 `libsparse_matmul.so`
4. **安装**：将 `.so` 拷贝到包目录下

### C 源码架构

Host 侧入口函数 `run_sparse_matmul` 的调用链：

1. **Tiling 生成**：根据 M/N/K 参数计算分块策略
2. **内存分配**：在 Device 侧分配 tiling 数据和 workspace
3. **Kernel 启动**：通过 `aclrtlaunch_matmul_sparse_custom` 启动 AscendC Kernel
4. **同步等待**：确保核函数完成后释放资源

## 适用场景

4:2 结构化稀疏矩阵乘法适用于：
- **NPU 量化推理**：INT8 权重经过 4:2 稀疏剪枝后的推理加速
- **MoE 模型优化**：Expert 权重的稀疏化部署
- **与 NPUSlim 量化流程协同**：在 INT8 量化后进一步应用稀疏化

## 代码示例

```python
import torch
from npuslim.ops.sparse_matmul import sparse_matmul_4to2

# 假设已完成 4:2 稀疏压缩
a = torch.randint(-128, 127, (512, 1024), dtype=torch.int8, device="npu:0")
b_dense = torch.randint(-128, 127, (768, 512), dtype=torch.int8, device="npu:0")
index = torch.zeros(768, 128, dtype=torch.uint8, device="npu:0")

# 执行稀疏矩阵乘法
c = sparse_matmul_4to2(a, b_dense, index)
print(c.shape)  # torch.Size([512, 768])
```
