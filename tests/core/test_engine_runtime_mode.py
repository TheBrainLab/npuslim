from unittest.mock import Mock

from npuslim.config.parser import RecipeTaskConfig, TaskExecutionConfig
from npuslim.core.engine import SlimEngine


def test_create_task_applies_execution_mode_to_model(monkeypatch):
    engine = SlimEngine.__new__(SlimEngine)
    model = Mock()
    engine.resources = {"qwen3": model}

    task_cfg = RecipeTaskConfig(
        name="quant",
        type="QuantizeTask",
        model="@qwen3",
        execution=TaskExecutionConfig(mode="streaming", chunk_size=2),
    )

    create_mock = Mock(return_value=object())
    monkeypatch.setattr("npuslim.core.engine.TaskRegistry.create", create_mock)

    engine._create_task(task_cfg)

    model.configure_runtime.assert_called_once_with(mode="streaming", chunk_size=2)
    create_mock.assert_called_once()
