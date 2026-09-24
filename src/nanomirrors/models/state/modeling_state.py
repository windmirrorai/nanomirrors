import torch
import torch.nn as nn

from .configuration_state import StateConfig

from transformers import GPT2Config, GPT2Model, LlamaConfig, LlamaModel

class StateBidirectionalGPT2(nn.Module):

    def __init__(self, config: StateConfig):
        super().__init__()

        gpt_config = GPT2Config(
            vocab_size=10,  # dummy value since we are not using token embeddings
            bos_token_id=None,
            eos_token_id=None,
            n_embd=config.hidden_dim,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_inner=config.n_inner,
            n_positions=config.cell_set_size,
            resid_pdrop=config.resid_pdrop,
            embd_pdrop=config.embd_pdrop,
            attn_pdrop=config.attn_pdrop,
            is_decoder=False,
            use_cache=False, # no k/v cache
        )
        gpt_config.is_causal = False
        self.transformer = GPT2Model(gpt_config)
        
        # disable causal attention
        for block in self.transformer.h:
            block.attn.is_causal = False
        # no token / positional embeddings
        self.transformer.wte.weight.requires_grad = False
        self.transformer.wpe.weight.requires_grad = False
        with torch.no_grad():
            self.transformer.wte.weight.zero_()
            self.transformer.wpe.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor: # [B, S, D_h] -> [B, cell_set_size, D_h]
        hidden_state = self.transformer(
            inputs_embeds=x
        ).last_hidden_state
        return hidden_state

class NoRoPE(nn.Module):
    """
        NoRoPE disables the rotary positional embeddings by 
        returning ones and zeros for the cosine and sine components, 
        effectively removing positional information from the attention mechanism.
    """
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, x, position_ids=None):
        B, S, _ = x.shape
        shape = (B, S, self.head_dim)
        return x.new_ones(shape), x.new_zeros(shape)


class StateBidirectionalLlama(nn.Module):

    def __init__(self, config: StateConfig):
        super().__init__()

        if config.hidden_dim % config.n_attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_attention_heads")

        llama_config = LlamaConfig(
            vocab_size=10,  # unused: we supply inputs_embeds
            bos_token_id=None,
            eos_token_id=None,
            hidden_size=config.hidden_dim,
            num_hidden_layers=config.n_transformer_layers,
            num_attention_heads=config.n_attention_heads,
            num_key_value_heads=config.n_attention_heads,
            head_dim=config.hidden_dim // config.n_attention_heads,
            intermediate_size=config.transformer_intermediate_dim,
            max_position_embeddings=config.cell_set_size,
            hidden_act="silu",
            rms_norm_eps=config.transformer_norm_eps,
            attention_dropout=config.transformer_attention_dropout,
            use_cache=False,
        )
        llama_config.is_causal = False

        self.transformer = LlamaModel(llama_config)

        for layer in self.transformer.layers:
            layer.self_attn.is_causal = False

        self.transformer.rotary_emb = NoRoPE(llama_config.head_dim)

        self.transformer.embed_tokens.weight.requires_grad_(False)
        with torch.no_grad():
            self.transformer.embed_tokens.weight.zero_()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: [B, S, hidden_dim]
        attention_mask: [B, S], 1=valid cell, 0=padding
        returns: [B, S, hidden_dim]
        """
        return self.transformer(
            inputs_embeds=x,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state


class StateMLP(nn.Module):

    def __init__(self, config: StateConfig, input_dim: int, n_layers: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = config.hidden_dim
        self.n_layers = n_layers

        layers = []
        in_dim = input_dim
        for i in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.LayerNorm(config.hidden_dim))
            layers.append(nn.GELU())
            in_dim = config.hidden_dim

        layers.append(nn.Linear(in_dim, config.hidden_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class StateLatentToGeneDecoder(nn.Module):

    def __init__(self, config: StateConfig, n_layers: int = 4):
        super().__init__()
        self.config = config
        layers = []
        for i in range(n_layers - 1):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(nn.LayerNorm(config.hidden_dim, eps=config.latent_decoder_layernorm_eps))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config.latent_decoder_dropout))

        layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
        layers.append(nn.ReLU())

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


## If output_space = `gene`, output = self.project_out(z + input) 
## If output_space = `embedding`, output = self.project_out(z + input); gene_expr = self.gene_decoder(output)   # E -> G
class StateModel(nn.Module):

    def __init__(self, config: StateConfig):
        super().__init__()
        self.config = config

        self.basal_encoder = StateMLP(config, config.input_dim, config.n_encoder_layers)
        self.pert_encoder = StateMLP(config, config.pert_dim, config.n_encoder_layers)
        self.transformer_backbone = StateBidirectionalLlama(config)
        self.project_out = nn.Linear(config.hidden_dim, config.input_dim) # f_recon
        self.relu = nn.ReLU()

        if config.use_batch_encoder:
            self.batch_encoder = nn.Embedding(config.n_batches, config.hidden_dim)
        if config.output_space == "embedding":
            self.gene_decoder = StateLatentToGeneDecoder(config)

    def forward(self, 
                basal_exprs: torch.Tensor, 
                perts: torch.Tensor, 
                batch: torch.Tensor = None) -> torch.Tensor:
        B, S, D = basal_exprs.shape
        h_cell = self.basal_encoder(basal_exprs) # [B, S, hidden_dim]
        h_pert = self.pert_encoder(perts) # [B, hidden_dim]

        h = h_cell + h_pert[:, None, :]

        if batch is not None and hasattr(self, "batch_encoder"):
            h_batch = self.batch_encoder(batch) # [B, hidden_dim]
            h = h + h_batch[:, None, :]

        z = self.transformer_backbone(h) # [B, S, D]
        output = self.relu(self.project_out((z + h_cell)))

        if self.config.output_space == "embedding":
            gene_expr = self.gene_decoder(output)
            return output, gene_expr
        return output

    