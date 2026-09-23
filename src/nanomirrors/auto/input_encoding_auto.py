from .registry import resolve_class


class AutoInputEncoder:
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        encoder_class = resolve_class(model_name, "input_encoder")
        return encoder_class.from_pretrained(model_name=model_name, **kwargs)

class AutoInputEncoderForPerturbationResponsePrediction:
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        encoder_class = resolve_class(
            model_name, "input_encoder", task_name="perturbation_response_prediction"
        )
        return encoder_class.from_pretrained(model_name=model_name, **kwargs)

