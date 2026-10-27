import torch

from torch import nn
from transformers import BertForMaskedLM, BertConfig
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .configuration_geneformer import GeneformerConfig

class GeneformerModel(nn.Module):
    def __init__(self, config: GeneformerConfig):
        super().__init__()
        self.config = config
        self.bert = BertForMaskedLM(config.to_hf_config())

        print(f"Initialized Geneformer with {self.get_num_params()/1e6:.2f}M parameters.")
    
    def get_num_params(self) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, input_ids, attention_mask=None):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True, 
            output_attentions=False
        )

    def encode_cell(self, 
                    input_ids, 
                    attention_mask=None,
                    embedding_layer=-2,
                    pooling = "mean") -> torch.Tensor:
        outputs = self.forward(input_ids, attention_mask)
        h = outputs.hidden_states[embedding_layer] # default uses 2nd-to-last layer.

        if pooling == "mean":
            h = h.mean(dim=1) # [B, H]
        elif pooling == "max":
            h = h.max(dim=1).values # [B, H]
        else:
            raise ValueError(f"Unsupported pooling type: {pooling}")

        return h # [B, H]

    def encode_gene(self, input_ids, attention_mask=None, embedding_layer=-2) -> torch.Tensor:
        outputs = self.forward(input_ids, attention_mask)
        h = outputs.hidden_states[embedding_layer] # default uses 2nd-to-last layer.

        return h # [B, L, H]
    
    @classmethod
    def from_pretrained(cls, model_name="Geneformer-V2-104M"):
        assert model_name in ["Geneformer-V2-104M", "Geneformer-V1-10M",], f"Unsupported model type {model_name}."
        
        config = GeneformerConfig.from_pretrained(model_name.lower())
        model = cls(config)
        
        model_path = hf_hub_download(repo_id="ctheodoris/Geneformer", 
                                     filename=f"{model_name}/model.safetensors")
        state_dict = load_file(model_path, device="cpu")
        model.bert.load_state_dict(state_dict, strict=False)

        return model

class GeneformerForPerturbationResponsePrediction(nn.Module):
    def __init__(self, 
                 config: GeneformerConfig,
                 n_genes: int,
                 head_dropout: float = 0.1):
        super().__init__()

        self.config = config
        self.n_genes = n_genes

        self.geneformer = GeneformerModel(config)
        self.delta_head = nn.Sequential(
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.output_head = nn.Linear(config.hidden_size, self.n_genes)

    def forward(self, 
                ctrl_input_ids,
                ctrl_attention_mask=None,
                edited_input_ids=None,
                edited_attention_mask=None):
        ctrl_embeddings = self.geneformer.encode_cell(ctrl_input_ids, ctrl_attention_mask) # [B, H]
        edited_embeddings = self.geneformer.encode_cell(edited_input_ids, edited_attention_mask) # [B, H]

        delta_embeddings = torch.cat([ctrl_embeddings, edited_embeddings], dim=-1)
        delta_hidden = self.delta_head(delta_embeddings) # [B, H]
        output = self.output_head(delta_hidden + ctrl_embeddings) # [B, n_genes]

        return output

    @classmethod
    def from_pretrained(cls, n_genes: int, model_name="Geneformer-V2-104M"):
        base = GeneformerModel.from_pretrained(model_name)
        model = cls(base.config, n_genes=n_genes)
        model.geneformer = base

        print(
            f"Loaded Geneformer base weights, with {model.geneformer.get_num_params()/1e6:.2f}M parameters. "
            f"Delta head randomly initialized with {sum(p.numel() for p in model.delta_head.parameters())/1e6:.2f}M parameters."
        )

        return model
