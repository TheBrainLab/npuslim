# Tasks

## BaseTask

```{eval-rst}
.. autoclass:: npuslim.tasks.base_task.BaseTask
   :members:
   :undoc-members:
   :show-inheritance:
```

## Task Types

### CompressorTask

Post-training quantization task.

**File**: `src/npuslim/tasks/compressor_task.py`

```{eval-rst}
.. autoclass:: npuslim.tasks.compressor_task.CompressorTask
   :members:
   :undoc-members:
```

### EvalTask

Model evaluation task.

**File**: `src/npuslim/tasks/eval_task.py`

```{eval-rst}
.. autoclass:: npuslim.tasks.eval_task.EvalTask
   :members:
   :undoc-members:
```

### SaveTask

Model save/export task.

**File**: `src/npuslim/tasks/save_task.py`

```{eval-rst}
.. autoclass:: npuslim.tasks.save_task.SaveTask
   :members:
   :undoc-members:
```

### SpeculativeTask

Speculative decoding task.

**File**: `src/npuslim/tasks/speculative_task.py`

```{eval-rst}
.. autoclass:: npuslim.tasks.speculative_task.SpeculativeTask
   :members:
   :undoc-members:
```

## Task Configuration

Tasks are configured in the pipeline section:

```yaml
pipeline:
  - type: ptq              # Task type
    algo_name: INT8Dynamic  # Algorithm selector
    ignore_layers: []        # Layer patterns to skip

  - type: eval             # Evaluation task

  - type: save             # Save task
    save_dir: ./outputs
```

## Layer Pattern Matching

Use `ignore_layers` to skip specific modules:

```python
# Direct match
ignore_layers: ["lm_head"]

# Glob pattern
ignore_layers: ["output.*"]

# Regex pattern
ignore_layers: ["re:lm_head.*"]
```
