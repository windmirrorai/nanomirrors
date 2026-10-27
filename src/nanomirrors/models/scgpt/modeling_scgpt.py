import torch
import torch.nn as nn
import anndata as ad
import numpy as np

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .input_encoding_scgpt import scGPTTokenizer
from .configuration_scgpt import scGPTConfig

class scGPTAttnention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        assert self.n_embd % self.n_head == 0, "Embedding dimension must be divisible by number of heads"

        self.in_proj = nn.Linear(self.n_embd, self.n_embd * 3, bias=config.bias)
        self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        self.dropout_p = config.dropout
        self.dropout = nn.Dropout(self.dropout_p)
        
        self.attn_imp = "regular"
        if config.attention_imp == "flash":
            has_flash_attention = hasattr(torch.nn.functional, "scaled_dot_product_attention")
            if not has_flash_attention:
                print("WARNING: Flash attention is not available in this version of PyTorch. Flash Attention requires PyTorch >= 2.0")
            else:
                self.attn_imp = "flash"
        elif config.attention_imp == "linear":
            print("WARNING: Linear attention is not implemented yet. Defaulting to regular attention.")
    
    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.BoolTensor | None) -> torch.Tensor:
        B, T, D = x.size()
        head_dim = D // self.n_head

        qkv = self.in_proj(x) # (B, T, 3 * n_embd)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2) # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2) # (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2) # (B, n_head, T, head_dim)

        if self.attn_imp == "flash":
            bool_mask = ~src_key_padding_mask[:, None, None, :] if src_key_padding_mask is not None else None
            attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
                                                                           attn_mask=bool_mask,
                                                                           dropout_p=self.dropout_p if self.training else 0.0,
                                                                           is_causal=False)
        else:
            attn_bias = None
            if src_key_padding_mask is not None:
                attn_bias = torch.zeros(B, 1, 1, T, device=x.device, dtype=q.dtype) # (B, 1, 1, T)
                attn_bias = attn_bias.masked_fill(src_key_padding_mask[:, None, None, :], float("-inf")) # (B, 1, 1, T)
            
            attn_weights = q @ k.transpose(-2, -1) / (head_dim ** 0.5) # (B, n_head, T, T)
            if attn_bias is not None:
                attn_weights = attn_weights + attn_bias # (B, n_head, T, T)
            attn_weights = torch.softmax(attn_weights, dim=-1) # (B, n_head, T, T)
            attn_weights = self.dropout(attn_weights)
            attn_output = attn_weights @ v # (B, n_head, T, head_dim)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D) # (B, T, n_embd)
        attn_output = self.out_proj(attn_output) # (B, T, n_embd)

        return attn_output


class scGPTBlock(nn.Module):

    def __init__(self,config):
        super().__init__()
        n_embd = config.n_embd
        n_hidden = config.n_hidden
        self.norm_first = config.pre_norm

        self.self_attn = scGPTAttnention(config)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)
        
        # MLP block
        self.linear1 = nn.Linear(n_embd, n_hidden)
        self.act = nn.ReLU() if config.activation_fn == "relu" else nn.GELU()
        self.linear2 = nn.Linear(n_hidden, n_embd)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    
    def forward(self, x, src_key_padding_mask: torch.BoolTensor | None) -> torch.Tensor:
        if self.norm_first:
            x = x + self.self_attn(self.norm1(x), src_key_padding_mask)
            x = x + self._mlp(self.norm2(x))
        else:
            x = self.norm1(x + self.self_attn(x, src_key_padding_mask))
            x = self.norm2(x + self._mlp(x))
        
        return x

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout1(self.act(self.linear1(x))))
        return self.dropout2(x)


class scGPTGeneEncoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        vocab_size = config.vocab_size
        n_embd = config.n_embd
        padding_idx = config.pad_token_id

        self.embedding = nn.Embedding(
            vocab_size, n_embd, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)  # (B, T, n_embd)
        x = self.enc_norm(x)
        return x
    
class scGPTContinuousValueEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        dropout_p = config.dropout
        d_model = config.n_embd
        self.max_value = config.input_continuous_max_value

        self.dropout = nn.Dropout(p=dropout_p)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class scGPTModel(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.encoder = scGPTGeneEncoder(config)
        self.value_encoder = scGPTContinuousValueEncoder(config) # TODO: support categorical/scaling value encoders as well.
        self.pert_encoder = nn.Embedding(3, config.n_embd, padding_idx=0) # TODO: double check cuz 0 means unperturbed.

        self.transformer_encoder = nn.ModuleDict(dict(
            layers = nn.ModuleList([scGPTBlock(config) for _ in range(config.n_layer)]),
        ))

        print(f"Initialized scGPT with {self.get_num_params()/1e6:.2f}M parameters.")
    
    def get_num_params(self) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def forward(self, 
                gene_tokens: torch.Tensor, 
                exprs: torch.Tensor, 
                padding_mask: torch.BoolTensor | None = None,
                pert_labels: torch.Tensor | None = None,
                batch_labels: torch.Tensor | None = None) -> torch.Tensor:
        gene_embds = self.encoder(gene_tokens)
        value_embds = self.value_encoder(exprs)
        if self.config.input_embd == "continuous":
            input_embds = gene_embds + value_embds
            
            if pert_labels is not None:
                pert_embds = self.pert_encoder(pert_labels)
                input_embds = input_embds + pert_embds
        else:
            raise NotImplementedError(
                        f"input_emb_style='{self.config.input_embd}' not yet supported. "
                        f"All published scGPT checkpoints use 'continuous'."
                    )
        
        h = input_embds # [B, T, n_embd]
        for layer in self.transformer_encoder.layers:
            h = layer(h, padding_mask) # [B, T, n_embd]
        
        return h
    
    def encode_cell(self, 
                    gene_tokens: torch.Tensor, 
                    exprs: torch.Tensor, 
                    padding_mask: torch.BoolTensor | None = None, 
                    batch_labels: torch.Tensor | None = None) -> torch.Tensor:
        h = self.forward(gene_tokens, exprs, padding_mask, batch_labels) # [B, T, n_embd]
        cls_embd = h[:, 0, :]
        
        return cls_embd

    def encode_gene(self, 
                    gene_tokens: torch.Tensor, 
                    exprs: torch.Tensor, 
                    padding_mask: torch.BoolTensor | None = None, 
                    batch_labels: torch.Tensor | None = None) -> torch.Tensor:
        h = self.forward(gene_tokens, exprs, padding_mask, batch_labels) # [B, T, n_embd]
        
        return h
    
    @classmethod
    def from_pretrained(cls, model_type="scGPT_human"):
        assert model_type in ["scGPT_human"], f"Unsupported model type {model_type}."

        config = scGPTConfig()
        model = cls(config)
     
        model_path = hf_hub_download(repo_id="paradoxdan/nano-scGPT", filename="model.safetensors")
        state_dict = load_file(model_path, device="cpu")
        model.load_state_dict(state_dict, strict=False)

        return model

class scGPTForPerturbationResponsePrediction(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scgpt = scGPTModel(config)
        self.decoder = AffineExprDecoder(config.n_embd)

    def forward(self, 
                gene_tokens: torch.Tensor, 
                exprs: torch.Tensor, 
                padding_mask: torch.BoolTensor | None = None,
                pert_labels: torch.Tensor | None = None,
                batch_labels: torch.Tensor | None = None) -> torch.Tensor:
        h = self.scgpt(gene_tokens, exprs, padding_mask, pert_labels) # [B, T, n_embd]
        pred = self.decoder(h, exprs).squeeze(-1) # [B, T]
        
        return pred
    
    def predict(self, 
                perturbation: str, 
                ctrl_adata: ad.AnnData, 
                tokenizer: scGPTTokenizer,
                n_samples: int = 64,
                pert_delimiter: str = "+",
                control_condition: str = "ctrl") -> torch.Tensor:
        device = next(self.parameters()).device
        self.eval()
        self.to(device)

        gene_names = ctrl_adata.var['gene_symbol'].tolist()
        gene_ids = torch.tensor([
            tokenizer.vocab.get(g, tokenizer.vocab.get("<pad>"))   # OOV → <pad> token id
            for g in gene_names
        ]).repeat(n_samples, 1).to(device) # [n_samples, G]

        pert_genes = perturbation.split(pert_delimiter)
        pert_genes = set([g for g in pert_genes if g != control_condition])
        pert_labels = [g in pert_genes for g in gene_names]
        pert_labels = torch.tensor(pert_labels, dtype=torch.int64).repeat(n_samples, 1).to(device) # [n_samples, G]
        assert pert_labels.sum() > 0, f"Perturbation '{perturbation}' does not match any genes in the dataset."
        
        sampled_idx = np.random.choice(ctrl_adata.n_obs, size=n_samples, replace=True)
        sampled_ctrl_adata = ctrl_adata[sampled_idx, :].copy()
        gene_values = torch.tensor(sampled_ctrl_adata.X.toarray(), dtype=torch.float32).to(device)
        src_key_padding_mask = torch.zeros_like(gene_values, dtype=torch.bool).to(device)

        with torch.no_grad():
            pred = self.forward(gene_ids, gene_values, src_key_padding_mask, pert_labels) # [n_samples, G]

        return pred.mean(dim=0) # [G]

    @classmethod
    def from_pretrained(cls, n_genes: int, model_name="scGPT_human"):
        base = scGPTModel.from_pretrained(model_name)
        
        model = cls(base.config)
        model.scgpt = base

        print(
            f"Loaded scGPT base weights, with {model.scgpt.get_num_params()/1e6:.2f}M parameters. "
            f"Decoder randomly initialized with {sum(p.numel() for p in model.decoder.parameters())/1e6:.2f}M params."
        )
        return model



class AffineExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int
    ):
        
        super().__init__()

        self.coeff_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )

        self.bias_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        coeff = self.coeff_decoder(x).squeeze(-1) # [B, T]
        bias = self.bias_decoder(x).squeeze(-1) # [B, T]

        pred = coeff * values + bias
        return pred
