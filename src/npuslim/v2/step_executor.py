# src/npuslim/v2/step_executor.py
"""Step executor for running algorithm steps."""
from typing import Dict, List, Any
from loguru import logger

from npuslim.v2.algorithm import StepInfo
from npuslim.v2.context import AlgorithmContext


class StepExecutor:
    """
    Executes steps in order, managing intermediate data.
    Passes outputs to next step or streaming.
    """

    def __init__(self, context: AlgorithmContext, steps: List[StepInfo]):
        self.context = context
        self.steps = steps
        self.intermediates: Dict[str, Any] = {}

    def execute(self) -> Dict[str, Any]:
        """Execute all steps, return final outputs."""
        final_outputs = {}

        for step_info in self.steps:
            # Gather inputs
            inputs = self._gather_inputs(step_info.requires)

            # Execute step
            logger.debug(f"Executing step {step_info.method.__name__}")
            result = step_info.method(self.context, **inputs)

            if result is None:
                continue

            # Store outputs
            for key, value in result.items():
                if key in step_info.produces:
                    self.intermediates[key] = value
                else:
                    final_outputs[key] = value

            # Auto-emit produced outputs if streaming
            for key in step_info.produces:
                if self.context.is_streaming and key in ["quantized", "packed"]:
                    self.context.emit(
                        f"{self.context.current_layer_name}.{key}",
                        self.intermediates[key]
                    )

        return final_outputs

    def _gather_inputs(self, requires: List[str]) -> Dict[str, Any]:
        """Gather required inputs from context or intermediates."""
        inputs = {}
        for req in requires:
            if req == "layer":
                inputs[req] = self.context.current_layer
            elif req == "layer_name":
                inputs[req] = self.context.current_layer_name
            elif req == "calib_data":
                inputs[req] = self.context.dataloader
            elif req in self.intermediates:
                inputs[req] = self.intermediates[req]
            elif hasattr(self.context, req):
                inputs[req] = getattr(self.context, req)
            else:
                raise ValueError(f"Required input '{req}' not available")
        return inputs
