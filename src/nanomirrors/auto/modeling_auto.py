from .registry import resolve_class

from .registry import _CHECKPOINTS

class AutoModel:
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        model_class = resolve_class(model_name, "model")
        return model_class.from_pretrained(model_name, **kwargs)


class AutoModelForPerturbationResponsePrediction:
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        model_class = resolve_class(
            model_name, "model", task_name="perturbation_response_prediction"
        )
        return model_class.from_pretrained(model_name, **kwargs)


def list_models(task_name: str = None):
    return list(_CHECKPOINTS.keys())