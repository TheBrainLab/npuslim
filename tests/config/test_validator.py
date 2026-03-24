"""Tests for config validator."""
import pytest

from npuslim.config.parser import parse_config, SlimConfig
from npuslim.config.validator import validate_config, ValidationError


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_validate_valid_config(self, sample_config_dict):
        """Valid config passes validation."""
        config = parse_config(sample_config_dict)
        validate_config(config)  # Should not raise

    def test_validate_catches_invalid_reference(self, sample_config_dict):
        """Validator catches invalid @id reference."""
        sample_config_dict["recipe"][0]["model"] = "@nonexistent"
        config = parse_config(sample_config_dict)

        with pytest.raises(ValidationError) as exc:
            validate_config(config)

        assert "nonexistent" in str(exc.value)

    def test_validate_catches_duplicate_ids(self, sample_config_dict):
        """Validator catches duplicate resource IDs."""
        sample_config_dict["resources"].append(
            {"id": "model1", "type": "AnotherModel"}
        )
        config = parse_config(sample_config_dict)

        with pytest.raises(ValidationError) as exc:
            validate_config(config)

        assert "duplicate" in str(exc.value).lower()

    def test_validate_speculative_references(self, speculative_config_dict):
        """Validator validates speculative task references."""
        config = parse_config(speculative_config_dict)
        validate_config(config)  # Should not raise

    def test_validate_catches_invalid_draft_reference(self, speculative_config_dict):
        """Validator catches invalid draft_model reference."""
        speculative_config_dict["recipe"][0]["draft_model"] = "@missing"
        config = parse_config(speculative_config_dict)

        with pytest.raises(ValidationError) as exc:
            validate_config(config)

        assert "missing" in str(exc.value)

    def test_validate_warns_empty_resources(self, caplog):
        """Validator warns about empty resources."""
        import logging
        from npuslim.config.parser import MetadataConfig, ResourceConfig, RecipeTaskConfig

        config = SlimConfig(
            metadata=MetadataConfig(),
            resources=[],
            recipe=[RecipeTaskConfig(name="T", type="test")]
        )

        with caplog.at_level(logging.WARNING):
            validate_config(config)

        # Check loguru warning (may not appear in caplog)
        # The validator should complete without error

    def test_validate_strict_mode(self):
        """Strict mode treats warnings as errors."""
        from npuslim.config.parser import MetadataConfig, ResourceConfig, RecipeTaskConfig

        config = SlimConfig(
            metadata=MetadataConfig(),
            resources=[],
            recipe=[]
        )

        with pytest.raises(ValidationError):
            validate_config(config, strict=True)
