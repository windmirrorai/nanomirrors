import torch
import torch.nn as nn

from .configuration_state import StateConfig

from transformers import GPT2Config, GPT2Model

class StateSetGPT2(nn.Module):

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



# !NOTE: MLP described in the paper; the code does something diff.
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

        layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class StateLatentToGeneDecoder(nn.Module):

    def __init__(self, config: StateConfig):
        super().__init__()
        self.config = config
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


# Paper:
# h = h_cell + h_pert + h_batch
# h_out = h + transformer(h)

# Current repo:
# h = h_cell + h_pert + h_batch
# h_out = transformer(h) + h_cell

## input = H_cell + H_pert + H_batch (optional)
## z = self.transformer_backbone(input)
## If output_space = `gene`, output = self.project_out(z + input) 
## If output_space = `embedding`, output = self.project_out(z + input); gene_expr = self.gene_decoder(output)   # E -> G
class StateModel(nn.Module):

    def __init__(self, config: StateConfig):
        super().__init__()
        self.config = config

        self.basal_encoder = StateMLP(config, config.input_dim, config.n_encoder_layers)
        self.pert_encoder = StateMLP(config, config.pert_dim, config.n_encoder_layers)
        self.transformer_backbone = StateSetGPT2(config)
        self.project_out = nn.Linear(config.hidden_dim, config.input_dim) # f_recon

        if config.batch_encoder:
            self.batch_encoder = nn.Embedding(config.batch_size, config.hidden_dim)
        if config.output_space == "embedding":
            self.gene_decoder = StateLatentToGeneDecoder(config)

    def _forward(self, basal_exprs: torch.Tensor, perts: torch.Tensor) -> torch.Tensor:
        B, S, D = basal_exprs.shape
        h_cell = self.basal_encoder(basal_exprs.view(B * S, D)) # [B * S, hidden_dim]
        h_pert = self.pert_encoder(perts) # [B, hidden_dim]

        h = h_cell.view(B, S, -1) + h_pert[:, None, :]
        z = self.transformer_backbone(h) # [B, S, D]
        output = self.project_out((z + h).view(B * S, -1)).view(B, S, -1)

        if self.config.output_space == "embedding":
            gene_expr = self.gene_decoder(output)
            return output, gene_expr
        return output

    def forward(self, batch):
        return self._forward(
            basal_exprs=batch["basal_exprs"],
            perts=batch["perts"],
        )

    
