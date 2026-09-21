from dataclasses import dataclass, field
from typing import List, Dict

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download

@dataclass
class scGPTTokenizerConfig:
    model_type: str = "scGPT_human"

    special_tokens: List[str] = field(default_factory=lambda: ["<cls>", "<pad>", "<eoc>"])
    max_length: int = 1200

    pad_value: float = -2.0
    cls_value: float = -2.0

    sampling: bool = True # Whether to do sampling or simple truncation when len > max_length.

    keep_first_n_tokens: int = 1 # do not bin the first n tokens, here specific refers to [CLS] token appended at the front.
    do_binning: bool = True
    num_bins: int = 51
    binning_deterministic: bool = False # whether deterministic or stochastic binning (i.e., randomize the bin assignment for values that fall on the edge of bins).


class scGPTTokenizer:

    def __init__(self, vocab: Dict[str, int], config):
        self.vocab = vocab
        self.special_tokens = config.special_tokens
        self.max_length = config.max_length
        self.pad_value = config.pad_value
        self.cls_value = config.cls_value
        self.sampling = config.sampling
        self.keep_first_n_tokens = config.keep_first_n_tokens
        self.do_binning = config.do_binning
        self.num_bins = config.num_bins
        self.binning_deterministic = config.binning_deterministic
        

    def encode(self, 
                 exprs: np.ndarray, # shape [B, G]
                 gene_names: List[str], # [G]
                 **kwargs) -> Dict[str, torch.Tensor]:
        # Align Vocab.
        vocab_genes_idx = [idx for idx, g in enumerate(gene_names) if g in self.vocab]

        aligned_exprs = exprs[:, vocab_genes_idx] # shape [B, G_vocab]
        aligned_gene_ids = np.array([self.vocab[gene_names[idx]] for idx in vocab_genes_idx]) # shape [G_vocab]
        if len(aligned_gene_ids) == 0:
            raise ValueError("None of the input genes are in the vocabulary.")
        print(f"Original genes: {len(gene_names)}| Genes in vocab: {len(aligned_gene_ids)}")

        # Determine the max sequence length T for this batch (after filtering zero-expression genes)
        max_expressed = np.max(np.sum(aligned_exprs > 0, axis=1)) + 1 # +1 for the [CLS] token
        T = min(max_expressed, self.max_length)
        print(f"Max expressed genes in this batch (after filtering zero-expression genes): {max_expressed-1}. Using T={T} for encoding.")

        encoded_cells = [self._encode(aligned_expr, aligned_gene_ids, T) for aligned_expr in aligned_exprs]

        # Pad and stack the cells into tensors of shape [B, T]
        out_gene_ids = torch.stack([cell["gene_ids"] for cell in encoded_cells])
        out_exprs = torch.stack([cell["exprs"] for cell in encoded_cells])
        out_padding_mask = torch.stack([cell["padding_mask"] for cell in encoded_cells])

        return {
            "gene_ids": out_gene_ids,
            "exprs": out_exprs,
            "padding_mask": out_padding_mask,
        }
    
    def _encode(self, exprs: np.ndarray, gene_ids: np.ndarray, T: int) -> Dict[str, torch.Tensor]:
        """
        Encode a single cell into gene_ids, exprs, and padding_mask.

        Args:
            exprs (`np.ndarray` of shape `[G_vocab]`): 
                the expression values of the genes in the vocab for this cell.
            gene_ids (`np.ndarray` of shape `[G_vocab]`): 
                the gene IDs of the genes in the vocab for this cell.
            T (`int`):
                the max sequence length for the input batch. The output gene_ids and exprs will be padded/trimmed to this length.
        Returns:
            A dict containing:
            - `gene_ids` (`torch.Tensor` of shape `[T]`): the gene IDs of the tokens for this cell.
            - `exprs` (`torch.Tensor` of shape `[T]`): the expression values of the tokens for this cell.
            - `padding_mask` (`torch.Tensor` of shape `[T]`): a boolean mask indicating which tokens are padding (True for padding, False for real tokens).
        """
        # Non-zero genes & expressions
        non_zero_mask = exprs > 0
        gene_ids = gene_ids[non_zero_mask]
        exprs = exprs[non_zero_mask]

        # Append [CLS] token at the front
        gene_ids = np.concatenate(([self.vocab["<cls>"]], gene_ids))
        exprs = np.concatenate(([self.cls_value], exprs))

        if self.do_binning:
            bins = np.quantile(exprs[self.keep_first_n_tokens:], q=np.linspace(0, 1, self.num_bins-1))
            binned_exprs = self._digitize(exprs[self.keep_first_n_tokens:], bins)
            exprs = np.concatenate((exprs[:self.keep_first_n_tokens], binned_exprs))
        
        out_gene_ids = np.full(T, self.vocab["<pad>"], dtype=np.int64)
        out_exprs = np.full(T, self.pad_value, dtype=np.float32)
        out_padding_mask = np.zeros(T, dtype=bool)

        if len(gene_ids) <= T:
            out_gene_ids[:len(gene_ids)] = gene_ids
            out_exprs[:len(exprs)] = exprs
            out_padding_mask[len(gene_ids):] = True
        else:
            if self.sampling:
                indices = torch.randperm(len(gene_ids) - self.keep_first_n_tokens)[: T - self.keep_first_n_tokens]
                indices = torch.cat([torch.arange(self.keep_first_n_tokens), indices + self.keep_first_n_tokens], dim=0)
                out_gene_ids = gene_ids[indices]
                out_exprs = exprs[indices]
            else:
                out_gene_ids = gene_ids[:T]
                out_exprs = exprs[:T]

        return {
            "gene_ids": torch.from_numpy(out_gene_ids).to(dtype=torch.long),
            "exprs": torch.from_numpy(out_exprs).to(dtype=torch.float32),
            "padding_mask": torch.from_numpy(out_padding_mask).to(dtype=torch.bool),
        }


    def _digitize(self, x: np.ndarray, bins: np.ndarray) -> np.ndarray:
        if self.binning_deterministic:
            return np.digitize(x, bins)   # left edge, no randomization
        left  = np.digitize(x, bins)
        right = np.digitize(x, bins, right=True)
        return np.ceil(np.random.rand(len(x)) * (right - left) + left).astype(np.int64)


    @classmethod
    def from_pretrained(cls, model_type="scGPT_human"):
        if model_type == "scGPT_human":
            config = scGPTTokenizerConfig()
            vocab_file = hf_hub_download(repo_id="wanglab/scGPT-human", filename="vocab.json")
        else:
            raise ValueError(
                f"Unsupported model_type: {model_type}. Supported types: ['scGPT_human']."
            )

        with open(vocab_file, "r") as f:
            vocab = json.load(f)
        for s in config.special_tokens:
            if s not in vocab:
                vocab[s] = len(vocab)
        
        return cls(vocab, config)


class scGPTInputEncoder:
    """
    Encode a generic perturbation batch into scGPT model inputs.

    Generic batch:
        exprs:             [B, G]
        ctrl_exprs:        [B, n_ctrl, G]
        perturbations:     list[str], len B
        is_ctrl:           bool [B]

    scGPT inputs:
        gene_ids:               [B, G]
        gene_values:            [B, G]
        src_key_padding_mask:   [B, G]
        pert_labels:            [B, G]
    """

    def __init__(self, 
                 gene_names: List[str], 
                 tokenizer_type="scGPT_human", 
                 perturbations_separator="+",
                 truncate_to_max_length=False,
                 keep_per_cell=True):
        self.tokenizer = scGPTTokenizer.from_pretrained(tokenizer_type)
        self.gene_names = gene_names
        self.T = self.tokenizer.max_length
        self.perturbations_separator = perturbations_separator
        self.truncate_to_max_length = truncate_to_max_length
        self.keep_genes_per_cell = keep_per_cell

        # fixed gene set for all cells.
        self.gene_ids = np.array([
            self.tokenizer.vocab.get(g, self.tokenizer.vocab.get("<pad>"))   # OOV → <pad> token id
            for g in self.gene_names
        ])

    def encode(self, batch: dict) -> dict:
        import pdb; pdb.set_trace()
        pert_names = [p.split(self.perturbations_separator) for p in batch["perturbations"]]
        pert_labels = [np.isin(self.gene_names, ps).astype(np.int64) for ps in pert_names]

        target_values = batch["exprs"]
        gene_values = batch["ctrl_exprs"]
        B, n_ctrl, n_genes = gene_values.shape

        # sample gene subset ONCE for the whole batch, only for train. for val/test, full gene sets are returned.
        # ![NOTE]: In the og scGPT tutorial, random sampling is performed and it means perturbed genes may get dropped with probability ((n_genes-max_length)/n_genes), 
        # which flags the whole pair as NOT perturbed but in reality it is. This can be misleading for the model.
        # A potential solution is to always keep the perturbed genes and only sample from the non-perturbed genes to fill up the T tokens.
        if self.truncate_to_max_length and n_genes > self.T:
            if self.keep_genes_per_cell: # always keep perturbed genes for each cell.
                # per-example: keep this example's perturbed gene(s), sample the rest from its own non-perturbed genes
                idx_list = []
                for b in range(B):
                    pert_idx_b = torch.where(pert_labels[b] > 0)[0]
                    non_pert_idx_b = torch.where(pert_labels[b] == 0)[0]

                    num_non_pert_to_sample = self.T - len(pert_idx_b)
                    if num_non_pert_to_sample > 0:
                        sampled = non_pert_idx_b[torch.randperm(len(non_pert_idx_b))[:num_non_pert_to_sample]]
                        idx_b = torch.cat([pert_idx_b, sampled])
                    else:
                        idx_b = pert_idx_b[:self.T]

                    idx_list.append(idx_b)
                idx = torch.stack(idx_list)  # (B, T)

                gene_values   = torch.gather(gene_values.view(B*n_ctrl, n_genes), 1, idx).view(B, n_ctrl, self.T)
                pert_labels   = torch.gather(pert_labels, 1, idx)
                target_values = torch.gather(target_values, 1, idx)
                gene_ids      = torch.from_numpy(self.gene_ids).long()[idx]     # (B, T), per-row now
            else: # og scgpt version, perturbed genes may get dropped with probability ((n_genes-max_length)/n_genes)
                idx = torch.randperm(n_genes)[:self.T]

                gene_values   = gene_values[:, :, idx]
                pert_labels   = pert_labels[:, idx]
                target_values = target_values[:, idx]
                gene_ids      = torch.from_numpy(self.gene_ids[idx]).long().unsqueeze(0).repeat(B, 1)
        else:
            gene_ids      = torch.from_numpy(self.gene_ids).long().unsqueeze(0).repeat(B, 1)

        # all-False padding mask
        src_key_padding_mask = torch.zeros_like(gene_values, dtype=torch.bool) # (B, n_ctrl, T)

        return {
            "gene_ids":             gene_ids,
            "gene_values":          gene_values,
            "src_key_padding_mask": src_key_padding_mask,
            "pert_labels":          pert_labels,
            "target_values":        target_values,
            "perturbations":        batch["perturbations"],
        }