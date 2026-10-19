from dataclasses import dataclass
from typing import Literal


@dataclass
class StateConfig:
    model_type: str = "state"

    # dataset-dependent feature dimensions
    input_dim: int | None = None
    output_dim: int | None = None
    pert_dim: int | None = None # num of perturbastions if multihot embedding, 5120 if using ESM2 embeddings
    use_multi_hot: bool = True # if using multi-hot encoding for perturbations
    use_pert_features: bool = False # if using ESM2 embeddings, dim 5120

    output_space: Literal["embedding", "gene", "all"] = "gene"

    # Cell encoder, perturbation encoder, and output projection.
    n_encoder_layers: int = 1
    n_decoder_layers: int = 1
    activation_fn: Literal["gelu", "relu", "silu"] = "gelu"
    mlp_dropout: float = 0.0

    # LatentToGeneDecoder specific configuration.
    latent_decoder_dropout: float = 0.1
    latent_decoder_layernorm_eps: float = 1e-5

    cell_set_size: int = 256
    hidden_dim: int = 768

    transformer_backbone: Literal["llama", "gpt2"] = "llama"
    n_transformer_layers: int = 8
    n_attention_heads: int = 12
    transformer_intermediate_dim: int = 3072 # 4 * hidden_dim
    transformer_attention_dropout: float = 0.0
    transformer_norm_eps: float = 1e-6 # LayerNorm epsilon for GPT2, RMSNorm for LLaMA.

    bidirectional_attention: bool = True
    use_position_embeddings: bool = False

    # Optional experimental batch conditioning.
    use_batch_encoder: bool = False
    n_batches: int | None = None

    # Distribution matching objective.
    loss: Literal["energy", "sinkhorn"] = "energy"
    blur: float = 0.05  # Relevant to Sinkhorn loss;