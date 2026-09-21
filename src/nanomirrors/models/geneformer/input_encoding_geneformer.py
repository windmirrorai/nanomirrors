from dataclasses import dataclass
from typing import List, Dict, Literal, Tuple

import pickle
import numpy as np
import torch
from huggingface_hub import hf_hub_download

@dataclass
class GeneformerTokenizerConfig:
    model_type: str = "Geneformer_V2"

    special_token: bool = True        
    model_input_size: int = 4096     
    target_sum: float = 10000.0

class GeneformerTokenizer:
    """
    Rank-value encoder: align vocab -> median-normalized -> drop zero exprs -> descending rank;
    """

    def __init__(
        self,
        gene_token_dict: Dict[str, int],
        gene_median_dict: Dict[str, float],
        config: GeneformerTokenizerConfig,
    ):
        self.vocab = gene_token_dict
        self.gene_median_dict = gene_median_dict
        self.special_token = config.special_token
        self.model_input_size = config.model_input_size
        self.target_sum = config.target_sum
        self.max_T = config.model_input_size
        self.max_genes = self.model_input_size - (2 if self.special_token else 0)

        self.pad_token_id = gene_token_dict["<pad>"]
        self.cls_token_id = gene_token_dict.get("<cls>")
        self.eos_token_id = gene_token_dict.get("<eos>")

        self.token2gene = {v: k for k, v in gene_token_dict.items()}

    def encode(
        self,
        exprs: np.ndarray,      # [B, G] raw counts, or log1p-transformed and target-sum normalized.
        gene_names: List[str],  # [G] Ensembl IDs
    ) -> Dict[str, torch.Tensor]:
        """
        Rank-value encode a batch of cells.

        Returns:
            input_ids:      [B, T]  rank-encoded token IDs, padded
            attention_mask: [B, T]  1 = real, 0 = pad
        """
        B, _ = exprs.shape
        target_sum_normalized = False
        # Convert tensor to numpy array if it's a torch tensor
        if isinstance(exprs, torch.Tensor):
            exprs = exprs.cpu().numpy()
        # Make sure raw counts are provided (not normalized)
        if not np.issubdtype(exprs.dtype, np.integer):
            print("Input expression matrix is not in raw counts. Converting from log1p scale.")
            before_log1p = np.exp(exprs) - 1
            exprs = before_log1p
            target_sum_normalized = True
        
        # Align the vocab
        vocab_genes_idx = [idx for idx, g in enumerate(gene_names) if g in self.vocab]
        aligned_exprs = exprs[:, vocab_genes_idx] # shape [B, G_vocab]
        aligned_gene_tokens = np.array([self.vocab[gene_names[idx]] for idx in vocab_genes_idx]) # shape [G_vocab]
        if len(aligned_gene_tokens) == 0:
            raise ValueError("None of the input genes are in the vocabulary.")
        print(f"Original genes: {len(gene_names)}| Genes in vocab: {len(aligned_gene_tokens)}")

        # Normalize by target sum and then median of each gene
        if not target_sum_normalized:
            aligned_exprs = aligned_exprs / aligned_exprs.sum(axis=1, keepdims=True) * self.target_sum  # shape [B, G_vocab]
        medians = np.array([self.gene_median_dict[self.token2gene[t]] for t in aligned_gene_tokens])[None, :] # shape [1,G_vocab]
        aligned_exprs = aligned_exprs / medians  # shape [B, G_vocab]

        # Rank by descending order within each cell
        desc_indices = np.argsort(-aligned_exprs, axis=1)[:, :self.max_genes]  # shape [B, G_vocab]
        input_ids = np.take_along_axis(aligned_gene_tokens[None, :].repeat(B, axis=0), desc_indices, axis=1) # shape [B, max_genes]

        if self.special_token:
            input_ids = np.concatenate(
                [
                    np.full((B, 1), self.cls_token_id, dtype=np.int64),
                    input_ids,
                    np.full((B, 1), self.eos_token_id, dtype=np.int64),
                ],
                axis=1,
            )

        attn_mask = (input_ids != self.pad_token_id).astype(np.int64)

        return {
            "input_ids": torch.from_numpy(input_ids).long(),
            "attention_mask": torch.from_numpy(attn_mask),
        }

    @classmethod
    def from_pretrained(cls, model_type="Geneformer_V2"):
        if model_type == "Geneformer_V2":
            config = GeneformerTokenizerConfig()
            token_file = "geneformer/token_dictionary_gc104M.pkl"
            median_file = "geneformer/gene_median_dictionary_gc104M.pkl"
        elif model_type == "Geneformer_V1":
            config = GeneformerTokenizerConfig(
                model_type="Geneformer_V1",
                special_token=False,
                model_input_size=2048,
            )
            token_file = "geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl"
            median_file = "geneformer/gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl"
        else:
            raise ValueError(
                f"Unsupported model_type: {model_type}. "
                f"Supported: ['Geneformer_V2', 'Geneformer_V1']"
            )

        token_path = hf_hub_download(
            repo_id="ctheodoris/Geneformer", filename=token_file
        )
        median_path = hf_hub_download(
            repo_id="ctheodoris/Geneformer", filename=median_file
        )

        with open(token_path, "rb") as f:
            gene_token_dict = pickle.load(f)
        with open(median_path, "rb") as f:
            gene_median_dict = pickle.load(f)

        return cls(gene_token_dict, gene_median_dict, config)


class GeneformerInputEncoder:
    """
    Encode a generic input batch into Geneformer model inputs.
    
    Generic batch:
        exprs:                [B, G]          cell gene expressions (raw counts, or prenormalized)
        
    Output:
        gene_tokens:            [B, T]            tokenized gene identifiers corresponding to columns in exprs
        attention_mask:         [B, T]            attention mask, 1 for valid tokens, 0 for padding.
    """

    def __init__(self, gene_names, tokenizer_type: str = "Geneformer_V2"):
        self.gene_names = gene_names
        self.tokenizer = GeneformerTokenizer.from_pretrained(tokenizer_type)

    def encode(self, batch: dict) -> dict:
        encoded_inputs = self.tokenizer.encode(batch["exprs"], self.gene_names)
        return {
            "input_ids": encoded_inputs["input_ids"],
            "attention_mask": encoded_inputs["attention_mask"],
        }