from dataclasses import dataclass

from transformers import BertConfig


@dataclass
class GeneformerConfig:
    model_type: str = "geneformer"

    max_position_embeddings: int = 4096
    vocab_size: int = 20275
    type_vocab_size: int = 2
    pad_token_id: int = 0

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_hidden_layers: int = 12

    hidden_act: str = "relu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    position_embedding_type: str = "absolute"

    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12

    @classmethod
    def from_pretrained(cls, name: str):
        configs = {
            "geneformer-v1-10m": dict(
                max_position_embeddings=2048,
                vocab_size=25426,
                hidden_size=256,
                intermediate_size=512,
                num_attention_heads=4,
                num_hidden_layers=6,
                hidden_dropout_prob=0.02,
                attention_probs_dropout_prob=0.02,
            ),
            "geneformer-v2-104m": dict(
                max_position_embeddings=4096,
                vocab_size=20275,
                hidden_size=768,
                intermediate_size=3072,
                num_attention_heads=12,
                num_hidden_layers=12,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
            ),
            "geneformer-v2-316m": dict(
                max_position_embeddings=4096,
                vocab_size=20275,
                hidden_size=1152,
                intermediate_size=4608,
                num_attention_heads=18,
                num_hidden_layers=18,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
            ),
        }

        if name not in configs:
            raise ValueError(
                f"Unknown Geneformer config: {name}. "
                f"Available: {list(configs)}"
            )

        return cls(**configs[name])

    def to_hf_config(self) -> BertConfig:
        return BertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            pad_token_id=self.pad_token_id,
            position_embedding_type=self.position_embedding_type,
        )