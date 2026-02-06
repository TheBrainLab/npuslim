from abc import ABC, abstractmethod
from typing import Dict, Any, List
from loguru import logger
import re
import fnmatch

from npuslim.utils.utils import create_or_update_dataclass


class BaseTask(ABC):
    """
    Abstract Base Class for all tasks in the NPUSlim pipeline.
    Handles configuration parsing and resource management.
    """

    ConfigClass = None

    def __init__(self, config: Dict[str, Any], resources: Dict[str, Any]):
        # Automatically parse dict configuration into a dataclass if ConfigClass is specified
        if self.ConfigClass and isinstance(config, dict):
            try:
                self.cfg = create_or_update_dataclass(self.ConfigClass, config)
            except Exception as e:
                raise ValueError(
                    f"Config parsing failed for task '{self.__class__.__name__}': {e}"
                )
        else:
            self.cfg = config

        # Store global resources shared across tasks
        self.resources = resources
        self.model = resources.get("main_model")
        self.dataloader = resources.get("dataloader")
        self.engine = resources.get("engine")

    @abstractmethod
    def execute(self):
        """
        Entry point for task execution. Must be implemented by subclasses.
        """
        pass

    def _resolve_layer_names(self, user_patterns: List[str]) -> List[str]:
        """
        Resolves specific layer names from the model based on user-provided patterns.
        Supports direct names, wildcard (glob) patterns, and regex.
        """
        if not self.model:
            return []

        # Extract all leaf modules (modules without children) from the model
        all_leaf_names = [
            n
            for n, m in self.model.model.named_modules()
            if len(list(m.children())) == 0 and n
        ]

        # Combine model-specific default skip layers with user-provided patterns
        model_defaults = getattr(self.model, "skip_layer_names", [])
        combined_patterns = set(model_defaults)

        if user_patterns:
            combined_patterns.update(user_patterns)

        if not combined_patterns:
            return []

        # Expand patterns (regex/glob) into actual module names found in the model
        final_names = self._expand_patterns(all_leaf_names, list(combined_patterns))

        # Log the resolved layers for visibility, limiting to the first 5 entries
        if final_names:
            display_list = final_names[:5]
            more_count = len(final_names) - 5
            display_str = "\n".join([f"    - {n}" for n in display_list])
            if more_count > 0:
                display_str += f"\n    - ... and {more_count} more."

            logger.info(
                f"[{self.__class__.__name__}] The following layers match the ignore patterns and will be skipped:\n{display_str}"
            )

        return final_names

    @staticmethod
    def _expand_patterns(all_module_names: list, patterns: list) -> list:
        """
        Utility to expand pattern strings into a sorted list of unique module names.
        Supports 're:' prefix for regular expressions.
        """
        expanded_set = set()
        for pattern in patterns:
            matched = []
            # Handle Regex patterns if prefixed with 're:'
            if pattern.startswith("re:"):
                regex_str = pattern[3:]
                try:
                    reg = re.compile(regex_str)
                    matched = [name for name in all_module_names if reg.fullmatch(name)]
                except re.error as e:
                    logger.error(f"Invalid regex pattern '{regex_str}': {e}")
                    continue
            else:
                # Handle direct matches or standard glob (fnmatch) patterns
                if pattern in all_module_names:
                    matched = [pattern]
                else:
                    matched = fnmatch.filter(all_module_names, pattern)

            if matched:
                expanded_set.update(matched)
            else:
                logger.debug(
                    f"Layer pattern '{pattern}' did not match any layers in the model."
                )

        return sorted(list(expanded_set))
