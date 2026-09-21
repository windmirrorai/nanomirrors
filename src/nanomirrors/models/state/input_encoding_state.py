#
# data = perturb_datasets.load_dataset('arc_replogle_nadig')
# train_data, val_data, test_data = perturb_datasets.split_dataset(data)
# dataset = perturb_datasets.PairedSetPerturbationDataset(data, ...)
# batch = next(iter(dataset))
# inputs = {model}InputEncoder.encode(batch)
# outputs = model(**inputs)
#

from typing import List

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

class StateInputEncoder:
    """
    Encode a generic perturbation batch into STATE model inputs.

    Generic batch:
        exprs:             [B, S, G]
        ctrl_exprs:        [B, S, G]
        perturbations:     list[str], len B
        other_covariates:  dict[str, list]

    STATE inputs:
        basal_exprs: [B, S, input_dim]
        perts:       [B, S, pert_dim]
        targets:     [B, S, input_dim]
    """

    def __init__(
        self,
        perturbation_vocab: list[str],
        input_key: str = "ctrl_exprs",
        target_key: str = "exprs",
        pert_key: str = "perturbations",
        other_covariates_key: str = "other_covariates",
        perturbations_separator: str = "+",
    ):
        self.perturbation2idx = {pert: i for i, pert in enumerate(perturbation_vocab)}

        self.perturbation_vocab = perturbation_vocab
        self.input_key = input_key
        self.target_key = target_key
        self.pert_key = pert_key
        self.other_covariates_key = other_covariates_key
        self.perturbations_separator = perturbations_separator

    def encode(self, batch: dict) -> dict:
        basal_exprs = batch[self.input_key]
        B, S, _ = basal_exprs.shape

        # perturbation label -> multi-hot
        perts_list = [pert.split(self.perturbations_separator) for pert in batch[self.pert_key]]
        perts_indices = [[self.perturbation2idx[p] for p in pert_list] for pert_list in perts_list]
        perts = _create_multi_hot(
            num_classes=len(self.perturbation_vocab),
            indices=perts_indices,
        )            
        
        inputs = {
            "basal_exprs": basal_exprs,
            "perts": perts,
            "targets": batch[self.target_key],
        }

        return inputs
