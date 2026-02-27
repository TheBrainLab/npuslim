# Datasets

## BaseDataset

```{eval-rst}
.. autoclass:: npuslim.dataset.base_dataset.BaseDataset
   :members:
   :undoc-members:
   :show-inheritance:
```

## Dataset Classes

### C4Dataset

C4 web text dataset for calibration.

```{eval-rst}
.. autoclass:: npuslim.dataset.c4_dataset.C4Dataset
   :members:
   :undoc-members:
```

### WikiText2Dataset

WikiText-2 dataset for calibration.

```{eval-rst}
.. autoclass:: npuslim.dataset.wikitext2_dataset.WikiText2Dataset
   :members:
   :undoc-members:
```

### MMLUDataset

MMLU (Massive Multitask Language Understanding) dataset for evaluation.

```{eval-rst}
.. autoclass:: npuslim.dataset.mmlu_dataset.MMLUDataset
   :members:
   :undoc-members:
```

### TextDataset

Custom text dataset from local files.

```{eval-rst}
.. autoclass:: npuslim.dataset.text_dataset.TextDataset
   :members:
   :undoc-members:
```

## Usage

```python
from npuslim import DatasetFactory
from npuslim.utils.config_parser import DatasetConfig

# Create dataset
config = DatasetConfig(
    type="C4Dataset",
    num_samples=256,
    max_seq_length=2048
)

dataset = DatasetFactory.create(
    processor=tokenizer,
    config=config
)

# Create dataloader
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=1,
    collate_fn=dataset.collate_fn
)
```

## Configuration

| Parameter | Type | Description |
|-----------|--------|-------------|
| `type` | str | Dataset factory identifier |
| `data_path` | str | Path to raw data (optional) |
| `num_samples` | int | Maximum samples to load |
| `max_seq_length` | int | Maximum sequence length |
| `device` | str | Device for tensors |
| `seed` | int | Random seed for sampling |
