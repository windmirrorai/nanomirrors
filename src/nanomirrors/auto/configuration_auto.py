from .registry import resolve_class


class AutoConfig:
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        config_class = resolve_class(model_name, "config")
        return config_class.from_pretrained(model_name, **kwargs)
