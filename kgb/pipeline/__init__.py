"""Knowledge Graph dynamic unified pipeline module.

Enables dynamically chained logic executions across processing contexts.
"""

from .context import PipelineContext
from .step import PipelineStep, register_step, get_step, list_available_steps
from .runner import PipelineRunner
from .config import load_pipeline_config, build_pipeline_from_config, list_pipeline_configs

# Explicitly load implementations to trigger their @register_step decorators.
# These will register the steps in the registry upon importing the `pipeline` module.
from .steps import extraction, augmentation, consolidation, export, checkpoint, converter, visualization

__all__ = [
    "PipelineContext",
    "PipelineStep",
    "register_step",
    "get_step",
    "list_available_steps",
    "PipelineRunner",
    "load_pipeline_config",
    "build_pipeline_from_config",
    "list_pipeline_configs",
]
