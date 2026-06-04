# 校准数据

量化算法（如 GPTQ、QuIP）需要一组代表性样本（即校准数据）来估计模型各层的权重分布与激活统计量。校准数据的质量和多样性直接影响量化后模型的精度。NPUSlim v2 提供了灵活的数据集抽象，支持在线流式加载与本地文件读取。

---

## 1 校准数据的作用

训练后量化（Post-Training Quantization, PTQ）不重新训练模型，而是通过少量校准样本模拟真实推理的数据分布：

- **GPTQ**：利用校准数据计算 Fisher 信息矩阵（Hessian）的近似值，逐层优化权重量化误差。
- **QuIP**：通过校准数据引导非相干变换（Incoherent Processing），降低量化噪声对关键方向的影响。
- **INT8Dynamic**：不需要校准数据——逐通道权重缩放因子可从权重本身直接计算。

通常 128~256 条样本即可满足大多数量化场景的校准需求。样本应尽量覆盖目标任务的领域分布。

---

## 2 BaseDataset 基类设计

所有校准数据集均继承自 `npuslim.datasets.base_dataset.BaseDataset`，该基类位于 `src/npuslim/datasets/base_dataset.py`。

### 2.1 构造函数

```python
class BaseDataset(Dataset):
    def __init__(
        self,
        *args,
        processor: "ProcessorMixin",   # HuggingFace tokenizer/processor
        device: str = "cpu",           # 张量设备
        num_samples: int = 256,        # 采样数量
        max_seq_length: int = 2048,    # 最大序列长度
        **kwargs,
    ):
```

| 参数 | 说明 |
|------|------|
| `processor` | HuggingFace tokenizer 或 processor 实例，用于文本编码 |
| `device` | 输出张量的设备（`cpu`、`cuda`、`npu`） |
| `num_samples` | 从数据源采样的数量上限 |
| `max_seq_length` | 每条样本截断后的序列长度 |

### 2.2 输出格式

每条样本为包含三个键的字典：

```python
{
    "input_ids": torch.Tensor,      # shape: [1, seq_len]
    "attention_mask": torch.Tensor,  # shape: [1, seq_len]，全 1
    "labels": torch.Tensor,          # shape: [1, seq_len]，左移后的标签
}
```

### 2.3 collate_fn

基类提供了静态方法 `collate_fn`，用于将样本列表合并为一个批次：

- `torch.Tensor` 类型：标量用 `torch.stack`，非标量用 `torch.cat`
- 数值类型：转为 `torch.tensor`
- 其他类型（如字符串）：保持为列表

```python
from npuslim.datasets.base_dataset import BaseDataset

dataloader = DataLoader(dataset, batch_size=4, collate_fn=BaseDataset.collate_fn)
```

---

## 3 C4 数据集

C4（Colossal Clean Crawled Corpus）是 GPTQ 和 QuIP 量化中最常用的校准数据集。NPUSlim 通过 `C4Dataset` 类实现了多级回退的加载策略，确保在各种网络环境下都能正常工作。

### 3.1 注册信息

| 属性 | 值 |
|------|-----|
| 注册名 | `C4` |
| 别名 | `C4Dataset` |
| 模块 | `npuslim.datasets.c4_dataset` |

### 3.2 配置方式

```yaml
resources:
  - id: calib_data
    type: C4Dataset
    num_samples: 128          # 采样数量，默认 256
    max_seq_length: 2048      # 最大序列长度，默认 2048
    seed: 0                   # 随机种子，默认 0
```

### 3.3 加载策略

`C4Dataset` 采用三级回退策略加载 C4 数据：

```
优先级 1: HuggingFace Hub 在线加载（streaming 模式）
    ↓ 失败
优先级 2: HuggingFace Hub 标准缓存加载（非 streaming）
    ↓ 失败
优先级 3: 本地 Arrow 缓存文件直接读取
    ↓ 全部失败
    抛出 RuntimeError
```

**优先级 1 和 2（在线/标准缓存）** 会尝试两种配置名（`en` 和 `default`）与两种模式（streaming / non-streaming）的组合，共最多 4 次尝试：

```python
for streaming in (True, False):
    for config_name in ("en", None):
        ds = load_dataset("allenai/c4", name=config_name, split="train", streaming=streaming)
```

Streaming 模式下数据按需从网络获取，不占用磁盘空间；标准缓存模式下 `datasets` 库会将数据缓存到本地。

**优先级 3（本地 Arrow 缓存）** 扫描以下路径查找已缓存的 Arrow 分片文件：

1. `$HF_DATASETS_CACHE` 环境变量指向的目录
2. `$HF_HOME/datasets` 目录
3. `~/.cache/huggingface/datasets` 默认缓存目录

搜索模式为 `allenai___c4/*/*/*/c4-train-*.arrow`，并优先使用最新修改时间的目录。

### 3.4 采样策略

加载到原始数据迭代器后，`C4Dataset` 使用以下策略采样：

1. 以 `seed` 初始化 `random` 随机数生成器
2. 逐条遍历数据，用 `processor` 对文本进行 tokenize
3. 跳过 token 长度不超过 `max_seq_length` 的短文本
4. 对满足长度要求的文本，在 `[0, seq_len - max_seq_length - 1]` 范围内随机选取起始位置，截取连续 `max_seq_length` 个 token
5. 构造 `input_ids`、`attention_mask` 和 `labels`（仅保留最后一个 token 作为标签）
6. 重复直到收集够 `num_samples` 条样本

这种随机截取策略确保了校准样本在长文本中的位置具有多样性，避免位置偏倚。

### 3.5 环境变量

| 变量 | 说明 |
|------|------|
| `HF_ENDPOINT` | HuggingFace 镜像地址，如 `https://hf-mirror.com` |
| `HF_DATASETS_CACHE` | 自定义 datasets 缓存目录 |
| `HF_HOME` | HuggingFace 根目录，默认 `~/.cache/huggingface` |

在国内网络环境下，建议设置镜像加速：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

---

## 4 Text 数据集

`TextDataset` 用于从本地文件加载校准数据，支持 JSONL 和 Parquet 两种格式。适用于已有领域专属校准语料的场景。

### 4.1 注册信息

| 属性 | 值 |
|------|-----|
| 注册名 | `Text` |
| 别名 | `TextDataset`、`text` |
| 模块 | `npuslim.datasets.text_dataset` |

### 4.2 配置方式

```yaml
resources:
  - id: local_data
    type: Text
    data_path: ./data/calibration.jsonl    # 必填：文件路径
    num_samples: 64
    max_seq_length: 4096
```

文件格式由 `data_path` 的扩展名自动推断：`.parquet` 走 Parquet 解析路径，其余均按 JSONL 处理。

### 4.3 JSONL 格式

JSONL 文件每行一个 JSON 对象，支持三种对话格式：

**格式 1：messages 字段（推荐）**

```jsonl
{"messages": [{"role": "user", "content": "解释量子纠缠"}, {"role": "assistant", "content": "量子纠缠是..."}]}
{"messages": [{"role": "system", "content": "你是一个助手"}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**格式 2：conversations 字段**

```jsonl
{"conversations": [{"from": "human", "value": "什么是深度学习?"}, {"from": "gpt", "value": "深度学习是..."}]}
```

**格式 3：input/output 字段**

```jsonl
{"input": "翻译以下句子", "output": "Here is the translation..."}
```

每种格式均支持可选的 `system_prompt` 或 `system` 字段，会自动插入为 `system` 角色消息。

### 4.4 Parquet 格式

Parquet 文件必须包含 `text` 列，可选包含 `labels` 列：

```
text (str): 原始文本内容
labels (list[int], 可选): 自定义标签 ID 序列
```

未提供 `labels` 时，自动将 `input_ids` 左移一位作为标签。

### 4.5 文本处理流程

`TextDataset` 的文本处理流程如下：

1. 解析每条数据为 `messages` 列表（统一为 `[{role, content}]` 格式）
2. 角色名归一化：`human` → `user`，`gpt` → `assistant`
3. 若 processor 支持 `apply_chat_template`，使用模型自带的聊天模板渲染文本
4. 否则使用简单拼接：`role: content\n`
5. 对包含思维链（`<think`/`</think >`）的数据启用专用模板
6. 使用 processor 进行 tokenize，截断到 `max_seq_length`
7. 构造标准输出字典

### 4.6 JSONL 配置示例

```yaml
resources:
  - id: jsonl_data
    type: Text
    data_path: ./data/sft_samples.jsonl
    num_samples: 128
    max_seq_length: 2048

recipe:
  - name: "GPTQ_Quantization"
    type: compressor
    model: "@model"
    dataloader:
      dataset: "@jsonl_data"
      batch_size: 1
    algorithm:
      type: GPTQ
      wbits: 4
      groupsize: 128
    ignore_layers: []
    execution:
      mode: streaming
      chunk_size: 4
    saver:
      type: StreamingHuggingFaceSaver
      save_dir: "./outputs"
```

### 4.7 Parquet 配置示例

```yaml
resources:
  - id: parquet_data
    type: Text
    data_path: ./data/wikipedia.parquet
    num_samples: 256
    max_seq_length: 4096
```

> Parquet 模式需要安装 `pyarrow`：`pip install pyarrow`

---

## 5 自定义数据集

当内置的 C4 和 Text 数据集不满足需求时，可以通过注册机制扩展自定义数据集。

### 5.1 实现步骤

1. 创建继承 `BaseDataset` 的子类
2. 在 `__init__` 中调用 `super().__init__()` 并传入 `processor`、`num_samples`、`max_seq_length` 等参数
3. 实现 `_load_data()` 方法，将处理后的样本追加到 `self.data` 列表
4. 使用 `@DatasetRegistry.register()` 装饰器注册

### 5.2 代码模板

```python
from npuslim.datasets.base_dataset import BaseDataset
from npuslim.core import DatasetRegistry

@DatasetRegistry.register("MyCustom")  # 注册名
class MyCustomDataset(BaseDataset):
    """自定义校准数据集示例。"""

    def __init__(self, *args, data_dir: str, split: str = "train", **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.split = split
        self._load_data()

    def _load_data(self):
        import json
        from pathlib import Path
        from loguru import logger

        files = sorted(Path(self.data_dir).glob(f"{self.split}_*.json"))
        count = 0
        for fpath in files:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if count >= self.num_samples:
                        break

                    item = json.loads(line)
                    text = item.get("text", "")
                    enc = self.processor(
                        text,
                        return_tensors="pt",
                        max_length=self.max_length,
                        truncation=True,
                    )
                    input_ids = enc["input_ids"]
                    if input_ids.shape[1] == 0:
                        continue

                    self.data.append({
                        "input_ids": input_ids.to(self.device),
                        "attention_mask": enc["attention_mask"].to(self.device),
                        "labels": input_ids.roll(-1, dims=-1).to(self.device),
                    })
                    count += 1

        logger.info(f"MyCustomDataset: loaded {len(self.data)} samples from {self.data_dir}")
```

### 5.3 配置中使用

```yaml
resources:
  - id: my_data
    type: MyCustom
    data_dir: /path/to/data
    split: train
    num_samples: 128
    max_seq_length: 2048
```

### 5.4 注册机制说明

NPUSlim 提供两种注册方式：

**即时注册**——导入时立即注册：

```python
@DatasetRegistry.register("MyCustom")
class MyCustomDataset(BaseDataset):
    ...
```

**延迟注册**——首次使用时才导入模块，减少启动时间：

```python
# 在 datasets/__init__.py 中
DatasetRegistry.register_lazy("MyCustom", ".my_custom_dataset", "MyCustomDataset")
```

注册名和别名均不区分大小写。通过 `DatasetRegistry.list()` 可查看所有已注册的数据集类型。

---

## 6 数据集选型建议

| 场景 | 推荐数据集 | 说明 |
|------|-----------|------|
| 通用量化 | C4 | 大规模网络语料，覆盖面广 |
| 特定领域量化 | Text（JSONL/Parquet） | 使用领域内语料可获得更优量化精度 |
| 快速验证 | C4（少量样本） | `num_samples: 32` 即可粗略评估 |
| 离线环境 | Text（本地文件） | 无需网络连接 |
| 高质量 SFT 模型 | Text（messages 格式） | 保持与训练数据分布一致 |

**样本数量**：GPTQ 论文建议 128 条校准样本即可在多数任务上获得良好效果。增加样本数通常带来边际收益递减。

**序列长度**：建议与模型实际推理时的上下文长度一致。过短的序列可能导致长上下文场景下的量化精度下降。
