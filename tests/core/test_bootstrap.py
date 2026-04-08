from pathlib import Path

from npuslim.config.schema import EngineConfig, MetadataConfig, RecipeTaskConfig
from npuslim.core.bootstrap import apply_saver_path_policy


def test_apply_saver_path_policy_derives_save_path_from_save_dir():
    cfg = EngineConfig(
        metadata=MetadataConfig(name="n", description="d"),
        resources=[],
        recipe=[
            RecipeTaskConfig(
                name="t",
                type="compressor",
                saver={"type": "StreamingHuggingFaceSaver", "save_dir": "./outputs"},
            )
        ],
    )

    apply_saver_path_policy(cfg, Path("configs/v2/qwen-gptq.yaml"))
    saver_cfg = cfg.recipe[0].saver
    assert isinstance(saver_cfg, dict)
    assert saver_cfg["save_path"] == str(Path("outputs") / "v2/qwen-gptq")


def test_apply_saver_path_policy_keeps_explicit_save_path():
    cfg = EngineConfig(
        metadata=MetadataConfig(name="n", description="d"),
        resources=[],
        recipe=[
            RecipeTaskConfig(
                name="t",
                type="compressor",
                saver={
                    "type": "StreamingHuggingFaceSaver",
                    "save_dir": "./outputs",
                    "save_path": "./explicit/final",
                },
            )
        ],
    )

    apply_saver_path_policy(cfg, Path("configs/v2/qwen-gptq.yaml"))
    saver_cfg = cfg.recipe[0].saver
    assert isinstance(saver_cfg, dict)
    assert saver_cfg["save_path"] == "./explicit/final"


def test_apply_saver_path_policy_strips_singular_config_prefix():
    cfg = EngineConfig(
        metadata=MetadataConfig(name="n", description="d"),
        resources=[],
        recipe=[
            RecipeTaskConfig(
                name="t",
                type="compressor",
                saver={"type": "StreamingHuggingFaceSaver", "save_dir": "./outputs"},
            )
        ],
    )

    apply_saver_path_policy(cfg, Path("config/exp/opt-int8.yaml"))
    saver_cfg = cfg.recipe[0].saver
    assert isinstance(saver_cfg, dict)
    assert saver_cfg["save_path"] == str(Path("outputs") / "exp/opt-int8")
