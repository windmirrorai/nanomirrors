"""
Registry for model families, checkpoints, and task-specific components.
Dynamic resolution of model classes, input encoders, and configurations 
based on checkpoint names and tasks.
"""
from dataclasses import dataclass
from importlib import import_module
from typing import Optional

@dataclass(frozen=True)
class ModelFamily:
    module: str
    config_class: str
    input_encoder_class: str
    model_class: str


_FAMILIES = {
    "scgpt": ModelFamily("scgpt", "scGPTConfig", "scGPTInputEncoder", "scGPTModel"),
    "geneformer": ModelFamily("geneformer", "GeneformerConfig", "GeneformerInputEncoder", "GeneformerModel"),
    "state": ModelFamily("state", "StateConfig", "StateInputEncoder", "StateModel"),
}

# mapping from checkpoint names to model families.
_CHECKPOINTS = {
    "scGPT_human": "scgpt",
    "Geneformer-V2-104M": "geneformer",
    "Geneformer-V1-10M": "geneformer",
    "State": "state",
}


'''
Task-specific model, input encoder, and trainer implementations.
'''
_TASK_MODELS = {
    "perturbation_response_prediction": {
        "scgpt": "scGPTForPerturbationResponsePrediction",
        "geneformer": "GeneformerForPerturbationResponsePrediction",
        "state": "StateForPerturbationResponsePrediction",
    },
}

_TASK_INPUT_ENCODERS = {
    "embedding": {
        "scgpt": "scGPTInputEncoder",
        "geneformer": "GeneformerInputEncoder",
        "state": "StateInputEncoder",
    },
    "perturbation_response_prediction": {
        "scgpt": "scGPTInputEncoderForPerturbation",
        "geneformer": "GeneformerInputEncoderForPerturbation",
    },
}

_TASK_TRAINER_CONFIGS = {
    "perturbation_response_prediction": {
        "scgpt": "scGPTTrainerConfig",
        "geneformer": "GeneformerTrainerConfig",
        "state": "StateTrainerConfig",
    },
}

_TASK_TRAINERS = {
    "perturbation_response_prediction": {
        "scgpt": "scGPTTrainerForPerturbationResponsePrediction",
        "geneformer": "GeneformerTrainerForPerturbationResponsePrediction",
        "state": "StateTrainerForPerturbationResponsePrediction",
    },
}

def resolve_class(model_name: str, 
                  component: str,
                  *, 
                  task_name: str | None = None):
    if model_name not in _CHECKPOINTS:
        raise ValueError(
            f"Unsupported checkpoint: {model_name}. "
            f"Available: {list(_CHECKPOINTS)}"
        )
    family_name = _CHECKPOINTS[model_name]
    family = _FAMILIES[family_name]

    if component == "config":
        module_name = f"configuration_{family.module}"
        class_name = family.config_class
    elif component == "input_encoder":
        module_name = f"input_encoding_{family.module}"

        if task_name is None:
            class_name = f"{family.input_encoder_class}"
        else:
            implementations = _TASK_INPUT_ENCODERS.get(task_name)
            if implementations is None:
                raise ValueError(
                    f"Unsupported input-encoder task: {task_name}. "
                    f"Available tasks: {list(_TASK_INPUT_ENCODERS.keys())}"
                )
            if family_name not in implementations:
                supported = [
                    name for name, family_key in _CHECKPOINTS.items()
                    if family_key in implementations
                ]
                raise ValueError(
                    f"Checkpoint {model_name} has no input encoder for task {task_name}. "
                    f"Supported checkpoints: {supported}"
                )
            class_name = implementations[family_name]
    elif component == "model":
        module_name = f"modeling_{family.module}"
        if task_name is None:
            class_name = family.model_class
        else:
            implementations = _TASK_MODELS.get(task_name, {})
            if family_name not in implementations:
                supported = [
                    name for name, family_key in _CHECKPOINTS.items()
                    if family_key in implementations
                ]
                raise ValueError(
                    f"Checkpoint {model_name} does not support task {task_name}. "
                    f"Supported checkpoints: {supported}"
                )
            class_name = implementations[family_name]
    elif component == "trainer_config":
        module_name = f"training_{family.module}"
        if task_name is None:
            raise ValueError("Task name must be specified for trainer_config resolution.")
        implementations = _TASK_TRAINER_CONFIGS.get(task_name)
        if implementations is None:
            raise ValueError(
                f"Unsupported trainer_config task: {task_name}. "
                f"Available tasks: {list(_TASK_TRAINER_CONFIGS.keys())}"
            )
        if family_name not in implementations:
            supported = [
                name for name, family_key in _CHECKPOINTS.items()
                if family_key in implementations
            ]
            raise ValueError(
                f"Checkpoint {model_name} has no trainer_config for task {task_name}. "
                f"Supported checkpoints: {supported}"
            )
        class_name = implementations[family_name]
    elif component == "trainer":
        module_name = f"training_{family.module}"
        if task_name is None:
            raise ValueError("Task name must be specified for trainer resolution.")
        implementations = _TASK_TRAINERS.get(task_name)
        if implementations is None:
            raise ValueError(
                f"Unsupported trainer task: {task_name}. "
                f"Available tasks: {list(_TASK_TRAINERS.keys())}"
            )
        if family_name not in implementations:
            supported = [
                name for name, family_key in _CHECKPOINTS.items()
                if family_key in implementations
            ]
            raise ValueError(
                f"Checkpoint {model_name} has no trainer for task {task_name}. "
                f"Supported checkpoints: {supported}"
            )
        class_name = implementations[family_name]
    else:
        raise ValueError(f"Unknown component: {component}")

    module = import_module(
        f"..models.{family.module}.{module_name}", package=__package__
    )
    return getattr(module, class_name)
            