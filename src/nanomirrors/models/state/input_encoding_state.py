from typing import Dict, List

import torch

def _create_multi_hot(num_classes: int, indices: List[list]) -> torch.Tensor:
    """
        Create a multi-hot encoding tensor from a list of variable index lists.
    
        Args:
            num_classes (int): The number of classes for the multi-hot encoding.
            indices (List[list]): A list of lists containing the indices to be set as 1 in the multi-hot encoding.
        Returns:
            torch.Tensor: A multi-hot encoded tensor of shape [len(indices), num_classes].
    """
    max_len = max(len(idx_list) for idx_list in indices)
    padded_indices = [idx_list + [-1] * (max_len - len(idx_list)) for idx_list in indices]
    padded_indices = torch.tensor(padded_indices, dtype=torch.long)
    mask = padded_indices >= 0
    padded_indices.clamp_(min=0)

    multi_hot = torch.zeros(padded_indices.shape[0], num_classes)
    multi_hot.scatter_add_(dim=1, index=padded_indices, src=mask.float())

    return multi_hot

class StateInputEncoderForPerturbation:
    """
    Encode a generic set perturbation batch into STATE model inputs.

    Generic batch:
        exprs:             [B, S, G], assuming target_sum and log1p normalized.
        ctrl_exprs:        [B, S, G], assuming target_sum and log1p normalized.
        perturbations:     list[str], len B
        other_covariates:  dict[str, list]

    STATE inputs:
        basal_exprs:    [B, S, input_dim]
        perts:          [B, pert_dim] # since each set shares the same perturbation across all cells.
        targets:        [B, S, input_dim]
    """

    def __init__(
        self,
        perturbation_vocab: list[str] | None,
        perturbation_features: Dict[str, torch.Tensor] | None,
        use_multi_hot: bool = True, 
        use_pert_features: bool = False,
        input_key: str = "ctrl_exprs",
        target_key: str = "exprs",
        pert_key: str = "perturbations",
        other_covariates_key: str = "other_covariates",
        perturbations_separator: str = "+",
    ):
        if perturbation_vocab is None and perturbation_features is None:
            raise ValueError("Either perturbation_vocab or perturbation_features must be provided.")
        if use_multi_hot and perturbation_vocab is None:
            raise ValueError("perturbation_vocab must be provided if use_multi_hot is True.")
        if use_pert_features and perturbation_features is None:
            raise ValueError("perturbation_features must be provided if use_pert_features is True.")
        
        self.use_multi_hot = use_multi_hot
        self.use_pert_features = use_pert_features

        if use_multi_hot:
            self.perturbation2idx = {pert: i for i, pert in enumerate(perturbation_vocab)}
            self.perturbation_vocab = perturbation_vocab
        if use_pert_features:
            self.perturbation_features = perturbation_features
            self.unknown_perturbation_feature = torch.zeros_like(next(iter(perturbation_features.values())))
        
        self.input_key = input_key
        self.target_key = target_key
        self.pert_key = pert_key
        self.other_covariates_key = other_covariates_key
        self.perturbations_separator = perturbations_separator

    def encode(self, batch: dict) -> dict:
        basal_exprs = batch[self.input_key]
        B, S, _ = basal_exprs.shape

        # perturbation label -> multi-hot
        if self.use_multi_hot:
            perts_list = [pert.split(self.perturbations_separator) for pert in batch[self.pert_key]]
            perts_indices = [[self.perturbation2idx[p] for p in pert_list] for pert_list in perts_list]
            perts = _create_multi_hot(
                num_classes=len(self.perturbation_vocab),
                indices=perts_indices,
            )
        # perturbation label -> perturbation features using pretrained embeddings
        elif self.use_pert_features:
            perts_list = [pert.split(self.perturbations_separator) for pert in batch[self.pert_key]]
            merged_perts_list = []
            for ps in perts_list:
                if len(ps) > 1:
                    merged_perts = torch.stack([self.perturbation_features.get(p, self.unknown_perturbation_feature) for p in ps])
                    merged_perts = torch.mean(merged_perts, dim=0, keepdim=True)
                else:
                    merged_perts = self.perturbation_features.get(ps[0], self.unknown_perturbation_feature).view(1, -1)
                merged_perts_list.append(merged_perts)
            perts = torch.cat(merged_perts_list, dim=0) # [B, pert_dim]

        
        inputs = {
            "basal_exprs": basal_exprs,
            "perts": perts,
            "targets": batch[self.target_key],
        }

        return inputs
