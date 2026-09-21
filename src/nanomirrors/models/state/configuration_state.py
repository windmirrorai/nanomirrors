from dataclasses import dataclass


@dataclass
class StateConfig:
    model_type = "state"

    input_dim: int | None = None
    output_dim: int | None = None
    pert_dim: int | None = None
    embed_key: str = "X_hvg" # "X_hvg" | "emb" |
    output_space: str = "gene" # "embedding" | "gene" | "all"


    cell_set_size: int = 32
    hidden_dim: int = 128

    n_encoder_layers: int = 4 # Cell / Perturbation encoders
    n_decoder_layers: int = 4 # Output projection
    activation_fn: str = "gelu"
    dropout: float = 0.0

    # Transformer
    transformer_backbone: str = "gpt2" # "gpt2 | llama"
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 128
    n_inner: int = 512  # GPT2 FFN = 4 * hidden_dim
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0

    batch_encoder: bool = False

    predict_residual: bool = True # better naming
    loss: str = "energy"
    blur: float = 0.05