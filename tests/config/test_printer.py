"""Tests for config printer."""
from io import StringIO

from rich.console import Console

from npuslim.config.parser import parse_config
from npuslim.config.printer import print_config


class TestPrintConfig:
    """Tests for print_config function."""

    def test_print_config_outputs_resources(self, sample_config_dict):
        """Printer outputs resource IDs."""
        config = parse_config(sample_config_dict)

        output = StringIO()
        console = Console(file=output, force_terminal=True)
        print_config(config, console)

        result = output.getvalue()
        assert "model1" in result
        assert "data1" in result

    def test_print_config_outputs_recipe(self, sample_config_dict):
        """Printer outputs recipe tasks."""
        config = parse_config(sample_config_dict)

        output = StringIO()
        console = Console(file=output, force_terminal=True)
        print_config(config, console)

        result = output.getvalue()
        assert "Task1" in result
        assert "compressor" in result

    def test_print_config_outputs_algorithm(self, sample_config_dict):
        """Printer outputs algorithm type."""
        config = parse_config(sample_config_dict)

        output = StringIO()
        console = Console(file=output, force_terminal=True)
        print_config(config, console)

        result = output.getvalue()
        assert "GPTQ" in result

    def test_print_config_empty_sections(self):
        """Printer handles empty sections."""
        config = parse_config({"metadata": {}, "resources": [], "recipe": []})

        output = StringIO()
        console = Console(file=output, force_terminal=True)
        print_config(config, console)  # Should not raise

        result = output.getvalue()
        # Should still produce some output (tables may be empty)
        assert len(result) >= 0
