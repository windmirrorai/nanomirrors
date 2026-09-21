from dataclasses import dataclass


@dataclass
class scGPTConfig:
    model_type = "scGPT"

    vocab_size: int = 60697 # NOTE: humans, mice might have different vocab. 60694 genes + 3 special tokens (<cls>, <pad>, <eoc>) for human scGPT.
    n_positions: int = 1200
    n_embd: int = 512
    n_hidden: int = 512 # Usually n_hidden = 4 * n_embd, but for scGPT it's equal to n_embd.
    n_layer: int = 12
    n_head: int = 8
    dropout: float = 0.2
    bias: bool = True
    activation_fn: str = "relu" # NOTE: [og] Flash and regular transformer encoder defaults to relu, but somewhat linear attn encoder uses gelu?
    pre_norm: bool = False
    attention_imp: str = "flash" # "flash" or "regular" or "linear"
    domain_specific_batchnorm: bool = False
    

    pad_token: str = "<pad>"
    pad_token_id: int = 60694
    pad_value: int = -2 # TODO: double check if it should be set to 0 during prp. but it's never used in prp since usually no padding.

    use_batch_labels: bool = False
    num_batch_labels: int | None = None
    input_embd: str = "continuous" # "continuous" or "categorical" or "scaling"
    n_input_bins: int | None = None # only used if input_embd is "categorical"
    input_continuous_max_value: float = 512.0 # only used if input_embd is "continuous"
    explicit_zero_prob: bool = False # whether to explicitly model zero probability for each gene