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

    def encode(self, input_ids, attention_mask=None, embedding_layer=-2) -> torch.Tensor:
        outputs = self.forward(input_ids, attention_mask)
        h = outputs.hidden_states[embedding_layer] # default uses 2nd-to-last layer.

        return h # [B, L, H]
    
    @classmethod
    def from_pretrained(cls, model_type="Geneformer-V2-104M"):
        assert model_type in ["Geneformer-V2-104M", "Geneformer-V1-10M",], f"Unsupported model type {model_type}."
        
        config = GeneformerConfig.from_pretrained(model_type.lower())
        model = cls(config)
        
        model_path = hf_hub_download(repo_id="ctheodoris/Geneformer", 
                                     filename=f"{model_type}/model.safetensors")
        state_dict = load_file(model_path, device="cpu")
        model.load_state_dict(state_dict, strict=False)

        return model