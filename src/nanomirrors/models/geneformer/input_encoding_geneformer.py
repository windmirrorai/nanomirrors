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
        gene_symbol_to_ensembl_dict: Dict[str, str],
        config: GeneformerTokenizerConfig,
    ):
        self.vocab = gene_token_dict
        self.gene_median_dict = gene_median_dict
        self.ensembl_mapping_dict = ensembl_mapping_dict
        self.gene_symbol_to_ensembl_dict = gene_symbol_to_ensembl_dict
        self.special_token = config.special_token
        self.model_input_size = config.model_input_size
        self.target_sum = config.target_sum
        self.max_T = config.model_input_size
        self.max_genes = self.model_input_size - (2 if self.special_token else 0)

        self.pad_token_id = gene_token_dict["<pad>"]
        self.cls_token_id = gene_token_dict.get("<cls>")
        self.eos_token_id = gene_token_dict.get("<eos>")

        self.token2gene = {v: k for k, v in gene_token_dict.items()}

    def _canonical_gene(self, gene: str) -> str:
        """
        Get the canonical Ensembl ID for a given gene(ensembl ID or gene_symbol), stripping version suffixes if necessary.
        """
        gene = gene.strip()
        
        if not gene.startswith("ENS"):
            gene = self.gene_symbol_to_ensembl_dict.get(gene, gene)

        # Only strip Ensembl version suffixes, not dots in arbitrary labels.
        if gene.startswith("ENS") and gene.rpartition(".")[2].isdigit():
            gene = gene.rpartition(".")[0]
        
        return self.ensembl_mapping_dict.get(gene, gene)

    def _retain_genes(self, 
                      ids: List[int], 
                      retain_genes: List[str]) -> Tuple[List[int], List[str]]:
        """
        Ensure that the retained genes are included in the token IDs.
        Args:
            ids: List of token IDs for the current cell.
            retain_genes: List of retained genes in Ensembl ID format.
        Returns:
            ids: List of token IDs including the retained genes, if possible.
            uncovered_retained_genes: List of retained genes that were not found in the vocabulary.
        """
        uncovered = []
        genes_to_append = []
        retain_genes = list(dict.fromkeys(retain_genes))
        retain_tokens = []
        for rg in retain_genes:
            rg = self._canonical_gene(rg)
            
            rg_token = self.vocab.get(rg)
            if rg_token is None:
                if rg not in uncovered:
                    uncovered.append(rg)

            elif rg_token not in ids and rg_token not in genes_to_append:
                genes_to_append.append(rg_token)
            
            retain_tokens.append(rg_token)

        # remove enough non-retained genes from the end to make space for the retained genes
        if len(genes_to_append) > len(ids):
            raise ValueError("Not enough space to retain all specified genes.")
        
        removed_cnt = 0
        for pos in range(len(ids) - 1, -1, -1):
            if removed_cnt >= len(genes_to_append):
                break
            
            if ids[pos] not in retain_tokens:
                ids.pop(pos)
                removed_cnt += 1

        # Append the retained genes that were not already in the token IDs
        ids.extend(genes_to_append)

        return ids, uncovered
            

    def encode(
        self,
        exprs: np.ndarray,      # [B, G] raw counts, or log1p-transformed and target-sum normalized.
        gene_names: List[str],  # [G] Ensembl IDs
        retain_genes_list: List[List[str]] | None = None,
        input_log1ped: bool = True,
        target_sum_normalized: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Rank-value encode a batch of cells.
        Args:
            exprs: Expression matrix of shape [B, G].
            gene_names: List of gene names corresponding to the columns of exprs.
            retain_genes_list: List of genes to retain for each sample in the batch. Used for keeping perturbation genes in the input.
            input_log1ped: Whether the input exprs are log1p-transformed.
            target_sum_normalized: Whether to normalize the input exprs to the target sum.
        NOTE: the retrain_genes and gene_names both need to in Ensembl ID format instead of gene symbol.

        Returns:
            input_ids:      [B, T]  rank-encoded token IDs, padded
            attention_mask: [B, T]  1 = real, 0 = pad
            unknowns:       List of gene names that were not found in the vocabulary.
            uncovered_retained_genes: List of retained genes that were not found in the vocabulary.
        """
        if isinstance(exprs, torch.Tensor):
            exprs = exprs.cpu().numpy()

        # Make sure raw counts are provided (not normalized)
        if input_log1ped:
            # print("Input expression matrix is not in raw counts. Converting from log1p scale.")
            exprs = np.expm1(exprs)
        if not np.isfinite(exprs).all() or np.any(exprs < 0):
            raise ValueError("Expression values must be finite and nonnegative.")
        
        # Align the vocab: use ensembl mapping to sum multiple entries for the same gene;
        canonical_id_to_token = {}
        canonical_id_to_idx = {} # maps canonical Ensembl ID to its index in aligned_exprs/aligned_gene_tokens
        aligned_exprs = []
        aligned_gene_tokens = []
        unknowns = []
        uncovered_retained_genes = []

        for idx, g in enumerate(gene_names):
            if not g:
                continue

            canonical_ensembl_id = self._canonical_gene(g)
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

        # print(f"Original genes: {len(gene_names)}| Genes in vocab: {len(aligned_gene_tokens)}")

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

            if retain_genes_list is not None:
                ids, uncovered= self._retain_genes(ids, retain_genes_list[idx])
                if len(uncovered) > 0:
                    uncovered_retained_genes.extend(uncovered)

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
            "unknown_genes": set(unknowns),
            "uncovered_retained_genes": set(uncovered_retained_genes),
        }

    def edit(self, 
             input_ids: torch.Tensor, 
             attention_mask: torch.Tensor, 
             perturbations: List[List[str]],
             pert_method: str):
        """
        Edit the input_ids and attention_mask according to the specified perturbation(s).
        Args:
          -- `input_ids`: Tensor of input token IDs. shape [B, T]
          -- `attention_mask`: Tensor of attention masks. shape [B, T]
          -- `perturbations`: List of perturbation genes, of length  `B`.
          -- `pert_method`: Type of perturbation. One of ["activation", "interference", "deletion"].
        
        Returns:
          -- `edited_input_ids`: Edited tensor of input token IDs. shape [B, T]
          -- `edited_attention_mask`: Edited tensor of attention masks. shape [B, T]
        """
        assert pert_method in ["activation", "interference", "deletion"], \
        f"Unsupported perturbation type {pert_method}."

        edited_input_ids = input_ids.clone()
        edited_attention_mask = attention_mask.clone()

        for idx, pert_genes in enumerate(perturbations):
            if "control" in pert_genes: # edit does not apply to `control`.
                continue
                
            valid_ids = edited_input_ids[idx][edited_attention_mask[idx].bool()]
            input_genes = (
                valid_ids[1:-1].tolist()
                if self.special_token
                else valid_ids.tolist()
            )

            pert_tokens = []
            for gene in pert_genes:
                canonical_gene = self._canonical_gene(gene)
                if canonical_gene not in self.vocab:
                    continue

                token = self.vocab[canonical_gene]
                if token not in pert_tokens:
                    pert_tokens.append(token)

            missing = [t for t in pert_tokens if t not in input_genes]
            if missing:
                raise ValueError(
                    f"Cell {idx}: targets were not retained in baseline: {missing}"
                )

            # Assumes the baseline tokenizer already retained all in-vocab targets.
            remaining = [token for token in input_genes if token not in pert_tokens]

            if pert_method == "activation":
                input_genes = pert_tokens + remaining
            elif pert_method == "interference":
                input_genes = remaining + pert_tokens
            elif pert_method == "deletion":
                input_genes = remaining

            if self.special_token:
                input_genes = [
                    self.cls_token_id,
                    *input_genes,
                    self.eos_token_id,
                ]

            # Rewrite the row; deletion leaves additional padding.
            length = len(input_genes)
            if length > edited_input_ids.shape[1]:
                raise ValueError(f"Edited sequence exceeds allocated length with length {length} and allocated length {edited_input_ids.shape[1]}")

            edited_input_ids[idx].fill_(self.pad_token_id)
            edited_attention_mask[idx].zero_()

            edited_input_ids[idx, :length] = torch.as_tensor(
                input_genes,
                dtype=edited_input_ids.dtype,
                device=edited_input_ids.device,
            )
            edited_attention_mask[idx, :length] = 1
                    
        return edited_input_ids, edited_attention_mask


    @classmethod
    def from_pretrained(cls, model_name="Geneformer-V2-104M"):
        model_type = "Geneformer_V2" if "V2" in model_name else "Geneformer_V1"
        if model_type == "Geneformer_V2":
            config = GeneformerTokenizerConfig()
            ensembl_mapping_file = "geneformer/ensembl_mapping_dict_gc104M.pkl"
            token_file = "geneformer/token_dictionary_gc104M.pkl"
            median_file = "geneformer/gene_median_dictionary_gc104M.pkl"
            gene_symbol_to_ensembl_file = "geneformer/gene_name_id_dict_gc104M.pkl"
        elif model_type == "Geneformer_V1":
            config = GeneformerTokenizerConfig(
                model_type="Geneformer_V1",
                special_token=False,
                model_input_size=2048,
            )
            ensembl_mapping_file = "geneformer/gene_dictionaries_30m/ensembl_mapping_dict_gc30M.pkl"
            token_file = "geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl"
            median_file = "geneformer/gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl"
            gene_symbol_to_ensembl_file = "geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl"
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
        gene_symbol_to_ensembl_path = hf_hub_download(
            repo_id="ctheodoris/Geneformer", filename=gene_symbol_to_ensembl_file
        )

        with open(token_path, "rb") as f:
            gene_token_dict = pickle.load(f)
        with open(median_path, "rb") as f:
            gene_median_dict = pickle.load(f)
        with open(ensembl_mapping_path, "rb") as f:
            ensembl_mapping_dict = pickle.load(f)
        with open(gene_symbol_to_ensembl_path, "rb") as f:
            gene_symbol_to_ensembl_dict = pickle.load(f)

        return cls(gene_token_dict, gene_median_dict, ensembl_mapping_dict, gene_symbol_to_ensembl_dict, config)


class GeneformerInputEncoderForPerturbation:
    """
        Encode a paired perturbation input batch into Geneformer model inputs.
        
        batch:
            - ``exprs``: Target expression, shape ``(B, 1, n_genes)``.
            - ``ctrl_exprs``: Control expressions, shape ``(B, n_ctrl, n_genes)``.
            - ``perturbations``: List of perturbation names of length ``B``.
            - ``is_ctrl``: boolean array of length ``B`` indicating whether each target cell is a control.
            
        Output:
            ctrl_input_ids:            [B, T]            tokenized gene identifiers corresponding to columns in exprs
            ctrl_attention_mask:         [B, T]            attention mask, 1 for valid tokens, 0 for padding.
            edited_input_ids:            [B, T]            tokenized gene identifiers corresponding to columns in exprs
            edited_attention_mask:         [B, T]            attention mask, 1 for valid tokens, 0 for padding.
            targets:            [B, 1, n_genes]            target expression corresponding to columns in exprs
    
    """
    def __init__(self, 
                 gene_names: List[str],
                 perturbation_method: str, 
                 model_name: str = "Geneformer_V2"):
        self.gene_names = gene_names
        self.perturbation_method = perturbation_method
        self.tokenizer = GeneformerTokenizer.from_pretrained(model_name)

        self.unknown_genes = []
        self.uncovered_perturbations = []

    def encode(self, 
               batch: dict, 
               input_log1ped: bool = True, 
               target_sum_normalized: bool = True,
               perturbation_separator: str = "+",
               control_name: str = "ctrl") -> dict:
        pert_genes = batch.get("perturbations", None)
        pert_genes_list = []
        if pert_genes is not None:
            for pg in pert_genes:
                genes = pg.split(perturbation_separator)
                if control_name in genes and len(genes) > 1:
                    raise ValueError(f"Control gene '{control_name}' cannot be combined with other genes in perturbation '{pg}'.")
                elif control_name in genes:
                    pert_genes_list.append([]) # in case of control cell, empty list to indicate no specific genes are retained.
                else:
                    pert_genes_list.append(genes)
        B, n_ctrl, D = batch["ctrl_exprs"].shape
        assert n_ctrl == 1, "Currently only supports a single control per batch."

        encoded_ctrl = self.tokenizer.encode(batch["ctrl_exprs"].squeeze(1),
                                             self.gene_names,
                                             retain_genes_list=pert_genes_list,
                                             input_log1ped=input_log1ped, 
                                             target_sum_normalized=target_sum_normalized,
                                             )
        
        pert_method = None
        if self.perturbation_method == "CRISPRi":
            pert_method = "interference"
        elif self.perturbation_method == "CRISPRa":
            pert_method = "activation"
        else:
            pert_method = "deletion"

        edited_ctrl_input_ids, edited_ctrl_attention_mask = self.tokenizer.edit(
            encoded_ctrl["input_ids"], 
            encoded_ctrl["attention_mask"], 
            perturbations=pert_genes_list,
            pert_method=pert_method,
        )
        
        return {
            "ctrl_input_ids": encoded_ctrl["input_ids"],
            "ctrl_attention_mask": encoded_ctrl["attention_mask"],
            "edited_input_ids": edited_ctrl_input_ids,
            "edited_attention_mask": edited_ctrl_attention_mask,
            "targets": batch["exprs"].squeeze(1) # assuming batch["exprs"] has shape [B, 1, G]
        }


class GeneformerInputEncoder:
    """
    Encode a generic input batch into Geneformer model inputs.
    
    Generic batch:
        exprs:                [B, G]          cell gene expressions (raw counts, or prenormalized)
        
    Output:
        gene_tokens:            [B, T]            tokenized gene identifiers corresponding to columns in exprs
        attention_mask:         [B, T]            attention mask, 1 for valid tokens, 0 for padding.
    """

    def __init__(self, 
                 gene_names: List[str], 
                 model_name: str = "Geneformer_V2"):
        self.gene_names = gene_names
        self.tokenizer = GeneformerTokenizer.from_pretrained(model_name)

        print("Original gene names:", gene_names)

        if len(gene_names) > 0 and not gene_names[0].startswith("ENSG"):
            # convert to ensembl IDs using tokenizer's mapping
            self.gene_names = [self.tokenizer._canonical_gene(g) for g in gene_names]


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

    @classmethod
    def from_pretrained(cls, model_name: str, gene_names: list):
        return cls(gene_names=gene_names, model_name=model_name)