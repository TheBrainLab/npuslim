"""Config validator."""
from typing import List, Set
from loguru import logger

from npuslim.config.parser import EngineConfig


class ValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


def validate_config(config: EngineConfig, strict: bool = False) -> None:
    """
    Validate EngineConfig - check @id references exist.

    Args:
        config: Configuration to validate
        strict: If True, treat warnings as errors

    Raises:
        ValidationError: If validation fails
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Collect resource IDs
    resource_ids: Set[str] = set()
    seen_ids: Set[str] = set()

    for r in config.resources:
        if r.id in seen_ids:
            errors.append(f"Duplicate resource ID: '{r.id}'")
        seen_ids.add(r.id)
        resource_ids.add(r.id)

    # Validate recipe references
    for task in config.recipe:
        for ref_field in ["model", "data", "main_model", "draft_model"]:
            ref = getattr(task, ref_field)
            if ref:
                clean_ref = ref.lstrip("@")
                if clean_ref not in resource_ids:
                    errors.append(
                        f"Task '{task.name}': {ref_field}='{ref}' references non-existent resource"
                    )

    # Warnings
    if not config.resources:
        warnings.append("No resources defined")
    if not config.recipe:
        warnings.append("No recipe tasks defined")

    # Log warnings
    for w in warnings:
        logger.warning(f"Config validation: {w}")

    # Handle errors
    if strict and warnings:
        errors.extend(warnings)

    if errors:
        raise ValidationError("\n".join(errors))
