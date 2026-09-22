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
        ensembl_mapping_dict: Dict[str, str],
        config: GeneformerTokenizerConfig,
    ):
        self.vocab = gene_token_dict
        self.gene_median_dict = gene_median_dict
        self.ensembl_mapping_dict = ensembl_mapping_dict
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
        input_log1ped: bool = True,
        target_sum_normalized: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Rank-value encode a batch of cells.

        Returns:
            input_ids:      [B, T]  rank-encoded token IDs, padded
            attention_mask: [B, T]  1 = real, 0 = pad
        """
        if isinstance(exprs, torch.Tensor):
            exprs = exprs.cpu().numpy()

        # Make sure raw counts are provided (not normalized)
        if input_log1ped:
            print("Input expression matrix is not in raw counts. Converting from log1p scale.")
            exprs = np.expm1(exprs)
        if not np.isfinite(exprs).all() or np.any(exprs < 0):
            raise ValueError("Expression values must be finite and nonnegative.")
        
        # Align the vocab: use ensembl mapping to sum multiple entries for the same gene;
        canonical_id_to_token = {}
        canonical_id_to_idx = {} # maps canonical Ensembl ID to its index in aligned_exprs/aligned_gene_tokens
        aligned_exprs = []
        aligned_gene_tokens = []
        unknowns = []
        for idx, g in enumerate(gene_names):
            if not g:
                continue

            g = g.strip().split(".")[0]  # Remove version suffix if present in Ensembl ID
            canonical_ensembl_id = self.ensembl_mapping_dict.get(g)
            if canonical_ensembl_id not in self.vocab or canonical_ensembl_id not in self.gene_median_dict:
                unknowns.append(g)
                continue
            if canonical_ensembl_id not in canonical_id_to_token:
                canonical_id_to_token[canonical_ensembl_id] = self.vocab[canonical_ensembl_id]
                aligned_exprs.append(exprs[:, idx])
                aligned_gene_tokens.append(canonical_id_to_token[canonical_ensembl_id])
                canonical_id_to_idx[canonical_ensembl_id] = len(aligned_gene_tokens) - 1
            else:
                gene_idx = canonical_id_to_idx[canonical_ensembl_id]
                aligned_exprs[gene_idx] = aligned_exprs[gene_idx] + exprs[:, idx]
        if len(aligned_gene_tokens) == 0:
            raise ValueError("None of the input genes are in the vocabulary.")
        aligned_exprs = np.stack(aligned_exprs, axis=1)  # shape [B, G_vocab]
        aligned_gene_tokens = np.asarray(aligned_gene_tokens, dtype=np.int64)  # shape [G_vocab]

        print(f"Original genes: {len(gene_names)}| Genes in vocab: {len(aligned_gene_tokens)}")

        # Normalize by target sum and then median of each gene
        if not target_sum_normalized:
            total_counts = exprs.sum(axis=1, keepdims=True) # [B, 1] total counts per cell
            if np.any(total_counts <= 0):
                raise ValueError("Each cell must have positive total counts.")
            aligned_exprs = aligned_exprs / total_counts * self.target_sum  # shape [B, G_vocab]
        medians = np.array([self.gene_median_dict[self.token2gene[t]] for t in aligned_gene_tokens])[None, :] # shape [1,G_vocab]
        aligned_exprs = aligned_exprs / medians  # shape [B, G_vocab]

        # Rank by descending order within each cell
        cells = []
        max_len = 0
        for idx, row in enumerate(aligned_exprs):
            non_zero_mask = row != 0
            if sum(non_zero_mask) <= 0:
                print(f"[Warning] Skipping cell {idx} with no non-zero genes.")
                continue

            non_zero_tokens = aligned_gene_tokens[non_zero_mask]
            desc_indices = np.argsort(-row[non_zero_mask])[:self.max_genes]
            ids = non_zero_tokens[desc_indices].tolist()

            if self.special_token:
                ids = [self.cls_token_id] + ids + [self.eos_token_id]
            if len(ids) > max_len:
                max_len = len(ids)
            cells.append(ids)
        if len(cells) == 0:
            raise ValueError("No cells with non-zero genes found.")
        
        input_ids = np.full((len(cells), max_len), self.pad_token_id, dtype=np.int64)
        for idx, cell in enumerate(cells):
            input_ids[idx, :len(cell)] = cell

        attn_mask = (input_ids != self.pad_token_id).astype(np.int64)

        return {
            "input_ids": torch.from_numpy(input_ids).long(),
            "attention_mask": torch.from_numpy(attn_mask),
        }

    @classmethod
    def from_pretrained(cls, model_type="Geneformer_V2"):
        if model_type == "Geneformer_V2":
            config = GeneformerTokenizerConfig()
            ensembl_mapping_file = "geneformer/ensembl_mapping_dict_gc104M.pkl"
            token_file = "geneformer/token_dictionary_gc104M.pkl"
            median_file = "geneformer/gene_median_dictionary_gc104M.pkl"
        elif model_type == "Geneformer_V1":
            config = GeneformerTokenizerConfig(
                model_type="Geneformer_V1",
                special_token=False,
                model_input_size=2048,
            )
            ensembl_mapping_file = "geneformer/gene_dictionaries_30m/ensembl_mapping_dict_gc30M.pkl"
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
        ensembl_mapping_path = hf_hub_download(
            repo_id="ctheodoris/Geneformer", filename=ensembl_mapping_file
        )

        with open(token_path, "rb") as f:
            gene_token_dict = pickle.load(f)
        with open(median_path, "rb") as f:
            gene_median_dict = pickle.load(f)
        with open(ensembl_mapping_path, "rb") as f:
            ensembl_mapping_dict = pickle.load(f)

        return cls(gene_token_dict, gene_median_dict, ensembl_mapping_dict, config)


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

    def encode(self, 
               batch: dict, 
               input_log1ped: bool = True, 
               target_sum_normalized: bool = True) -> dict:
        encoded_inputs = self.tokenizer.encode(batch["exprs"], 
                                               self.gene_names, 
                                               input_log1ped=input_log1ped, 
                                               target_sum_normalized=target_sum_normalized)
        return {
            "input_ids": encoded_inputs["input_ids"],
            "attention_mask": encoded_inputs["attention_mask"],
        }