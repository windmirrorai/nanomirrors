import numpy as np
import anndata as ad
import scipy.sparse as sp
from torch.utils.data import Dataset

PERTURBATION_COL = "perturbation"
IS_CTRL_COL = "is_control"
CTRL_TOKEN = "control"
COMBO_SEP = "+"

GENE_SYMBOLS_COL = "gene_symbol"
GENE_ENSEMBL_IDS_COL = "gene_ensembl"

def _to_numpy(x, dtype=np.float32):
    if sp.issparse(x):
        x = x.toarray()

    return np.asarray(x, dtype=dtype)


class PerturbationDataset(Dataset):
    """
    A PyTorch Dataset for iterating each cell in perturbation data.
    Returns:
        - ``exprs``: Expression values for the cell, shape ``(n_genes,)``.
        - ``perturbations``: Perturbation applied to the cell.
        - ``is_ctrl``: Whether the cell is a control.
    """
    def __init__(self, adata):
        self.adata = adata

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        perturbation = self.adata.obs[PERTURBATION_COL].iloc[idx]

        return {
            "exprs": _to_numpy(self.adata.X[idx]).ravel(),
            "perturbations": perturbation,
            "is_ctrl": perturbation == CTRL_TOKEN,
        }


class PairedPerturbationDataset(Dataset):
    """
    Dynamically pair individual cells with sampled control cells.

    Args:
        adata: AnnData containing expression and perturbation annotations.
        strategy: Control sampling strategy. Currently supports ``"random"``.
        n_ctrl: Number of control cells sampled per target cell.
        include_ctrl_ctrl_pairs: Whether control cells can also be target cells.
        is_control_col: Name of the column in ``adata.obs`` containing is control annotations; defaults to ``is_control``.
        pair_covariates: List of observation columns for grouping cells, so that cells of the same pair share the same covariates;
          for example, pair covariates could be ``["cell_type"]``, so cell pairings are made within the same cell type.
        random_seed: Random seed for reproducibility.

    Returns:
        dict with:
            - ``exprs``: Target expression, shape ``(1, n_genes)``.
            - ``ctrl_exprs``: Control expressions, shape ``(n_ctrl, n_genes)``, sharing the same covariates as the target cell.
            - ``perturbations``: List of perturbation names.
            - ``covariates``: Dictionary of covariate key, values for the paired cells.
            - ``is_ctrl``: Whether the target cell is a control.

    Notes:
        Control cells are sampled on the fly, so pairings may change across epochs.
    """

    def __init__(self, 
                 adata: ad.AnnData,
                 strategy: str ="random", 
                 n_ctrl: int =1, 
                 include_ctrl_ctrl_pairs: bool = True,
                 perturbation_col: str = PERTURBATION_COL,
                 is_control_col: str = IS_CTRL_COL,
                 pair_covariates: list | None = None,
                 random_seed: int = 1,):
        self.adata = adata
        self.strategy = strategy
        self.n_ctrl = n_ctrl
        self.include_ctrl_ctrl_pairs = include_ctrl_ctrl_pairs
        self.perturbation_col = perturbation_col
        self.is_control_col = is_control_col
        self.pair_covariates = pair_covariates
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)


        # make sure pair_covariates exist in adata.obs
        if self.pair_covariates is not None:
            for cov in self.pair_covariates:
                if cov not in adata.obs.columns:
                    raise ValueError(f"pair_covariate {cov} not found in adata.obs")
        
        self.ctrl_idx = np.flatnonzero(adata.obs[is_control_col].to_numpy())
        self.pert_idx = np.flatnonzero(~adata.obs[is_control_col].to_numpy())
        self.perturbations = adata.obs[perturbation_col].unique().tolist()

        if len(self.ctrl_idx) == 0:
            raise ValueError("No control cells found in the dataset.")

        if self.pair_covariates is not None and len(self.pair_covariates) > 0:
            self.covariate_ctrl_idxs = {} # Dict mapping covariate values to control pool indices of the same covariate values
            for covariate_values, group in adata.obs.groupby(self.pair_covariates):
                if not isinstance(covariate_values, tuple):
                    covariate_values = (covariate_values,)

                ctrl_group = group[group[self.is_control_col]]
                self.covariate_ctrl_idxs[covariate_values] = adata.obs_names.get_indexer(ctrl_group.index)
        

    def __len__(self):
        return self.adata.n_obs if self.include_ctrl_ctrl_pairs else len(self.pert_idx)

    def __str__(self):
        num_pairs = len(self) if self.include_ctrl_ctrl_pairs else len(self.pert_idx)
        return (
            super().__str__()
            + f" | num pairs: {num_pairs}"
            + f" | num perts: {len(self.perturbations)}"
            + f" | num ctrl cells: {len(self.ctrl_idx)}"
            + f" | num pert cells: {len(self.pert_idx)}"
            + f" | num genes: {self.adata.n_vars}"
        )

    def __getitem__(self, idx):
        if self.include_ctrl_ctrl_pairs:
            cell_idx = idx
        else:
            cell_idx = self.pert_idx[idx]

        covariate_values = None
        if self.pair_covariates is None or len(self.pair_covariates) == 0:
            ctrl_pool_idx = self.ctrl_idx
        else:
            covariate_values = self.adata.obs.iloc[cell_idx][self.pair_covariates]
            ctrl_pool_idx = self.covariate_ctrl_idxs[tuple(covariate_values)]
        if len(ctrl_pool_idx) == 0:
            raise ValueError(f"No control cells found for covariate values: {covariate_values.to_dict()}")

        ctrl_paired = None
        if self.strategy == "random":
            ctrl_paired = np.random.choice(ctrl_pool_idx, 
                                           size=self.n_ctrl, 
                                           replace=self.n_ctrl>len(ctrl_pool_idx))
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return {
            "exprs": _to_numpy(self.adata.X[cell_idx]),
            "ctrl_exprs": _to_numpy(self.adata.X[ctrl_paired]),
            "perturbations": self.adata.obs[self.perturbation_col].iloc[cell_idx],
            "covariates": covariate_values.to_dict() if covariate_values is not None else {},
            "is_ctrl": self.adata.obs[self.is_control_col].iloc[cell_idx],
        }

class PairedSetPerturbationDataset(Dataset):
    """
    Dynamically pair a set of covariate cells with a set of sampled control cells of the same covariate (except `perturbation_col`).

    Args:
        adata: AnnData containing expression and perturbation annotations.
        set_size: Number of cells in a set.
        include_ctrl_ctrl_pairs: Whether to include control-control pairs; defaults to ``True``.
        strategy: Control sampling strategy. Currently supports ``"random"``.
        perturbation_col: Name of the column in ``adata.obs`` containing perturbation annotations; defaults to ``perturbation``.
        is_control_col: Name of the column in ``adata.obs`` containing is control annotations; defaults to ``is_control``.
        set_covariates: List of observation columns for grouping target cells into sets,  
            must include ``PERTURBATION_COL``; defaults to ``[PERTURBATION_COL,]``.
        perturbation_separator: Separator for splitting perturbation names in ``perturbation_col``; defaults to ``COMBO_SEP``.
        random_seed: Random seed for reproducibility; defaults to 1.

    Returns:
        dict with:
            - ``exprs``: Target expression set, shape ``(set_size, n_genes)``.
            - ``ctrl_exprs``: Control expression set, shape ``(set_size, n_genes)``.
            - ``perturbations``: List of perturbation names shared by the target cells in the set
            - ``covariates``: Dict mapping non-perturbation covariate names to the values shared by cells in the set.
    Notes:
        Target cells within each set share the same values for all ``set_covariates``,
        and control cells are sampled from the same covariate group (except for ``perturbation_col``).

        Control cells are sampled on the fly, so pairings may change across epochs.
    """

    def __init__(self, 
                 adata: ad.AnnData,
                 set_size: int = 512,
                 include_ctrl_ctrl_pairs: bool = True,
                 strategy: str ="random", 
                 perturbation_col: str = PERTURBATION_COL,
                 is_control_col: str = IS_CTRL_COL,
                 set_covariates: list[str] | None = None,
                 perturbation_separator: str = COMBO_SEP,
                 random_seed: int = 1):
        self.adata = adata
        self.set_size = set_size
        self.include_ctrl_ctrl_pairs = include_ctrl_ctrl_pairs
        self.strategy = strategy
        self.perturbation_col = perturbation_col
        self.perturbation_separator = perturbation_separator
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)

        if set_covariates is None:
            set_covariates = [self.perturbation_col]
        elif self.perturbation_col not in set_covariates:
            set_covariates = list(set_covariates)
            set_covariates.append(self.perturbation_col)
        self.set_covariates = list(set_covariates)

        # make sure set_covariates exist in adata.obs
        for cov in self.set_covariates:
            if cov not in adata.obs.columns:
                raise ValueError(f"set_covariate {cov} not found in adata.obs")

        self.is_control_col = is_control_col
        self.ctrl_idx = np.flatnonzero(adata.obs[self.is_control_col].to_numpy())
        self.pert_idx = np.flatnonzero(~adata.obs[self.is_control_col].to_numpy())
        self.perturbations = adata.obs[self.perturbation_col].unique().tolist()

        if not self.include_ctrl_ctrl_pairs:
            pert_obs = adata.obs.iloc[self.pert_idx]
        else:
            pert_obs = adata.obs

        self.sets = [] # List of indices for each set of cells

        for _, group in pert_obs.groupby(self.set_covariates):
            indices = adata.obs_names.get_indexer(group.index)

            self.rng.shuffle(indices)

            for i in range(0, len(indices), self.set_size):
                cell_indices = indices[i:i + self.set_size]

                if len(cell_indices) < self.set_size:
                    n_pad = self.set_size - len(cell_indices)

                    padding = self.rng.choice(
                        cell_indices,
                        size=n_pad,
                        replace=True,
                    )

                    cell_indices = np.concatenate([
                        cell_indices,
                        padding,
                    ])

                self.sets.append(cell_indices)

        self.covariates = [c for c in self.set_covariates if c != self.perturbation_col]
        if len(self.covariates) > 0:
            self.covariate_ctrl_idxs = {} # Dict mapping covariate values to control pool indices of the same covariate values (except for perturbation_col)
            for covariate_values, group in adata.obs.groupby(self.covariates):
                if not isinstance(covariate_values, tuple):
                    covariate_values = (covariate_values,)

                ctrl_group = group[group[self.is_control_col]]
                self.covariate_ctrl_idxs[covariate_values] = adata.obs_names.get_indexer(ctrl_group.index)


    def __len__(self):
        return len(self.sets)

    def __str__(self):
        return (
            super().__str__()
            + f" | num sets: {len(self.sets)}"
            + f" | num perts: {len(self.perturbations)}"
            + f" | num ctrl cells: {len(self.ctrl_idx)}"
            + f" | num pert cells: {len(self.pert_idx)}"
            + f" | num genes: {self.adata.n_vars}"
            + f" | set size: {self.set_size}"
            + f" | set covariates: {self.set_covariates}"
        )

    def __getitem__(self, idx):
        cell_set_idx = self.sets[idx]
        covariate_values = self.adata.obs.iloc[cell_set_idx[0]][self.set_covariates]

        if len(self.covariates) == 0:
            ctrl_pool_idx = self.ctrl_idx
        else:
            ctrl_pool_idx = self.covariate_ctrl_idxs[tuple(covariate_values[self.covariates])]
        if len(ctrl_pool_idx) == 0:
            raise ValueError(f"No control cells found for covariate values: {covariate_values[self.covariates].to_dict()}")
        
        ctrl_paired = None
        if self.strategy == "random":
            ctrl_paired = self.rng.choice(ctrl_pool_idx, 
                                          size=self.set_size, 
                                          replace=self.set_size>len(ctrl_pool_idx))
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return {
            "exprs": _to_numpy(self.adata.X[cell_set_idx]),
            "ctrl_exprs": _to_numpy(self.adata.X[ctrl_paired]),
            "perturbations": covariate_values[self.perturbation_col],
            "covariates": covariate_values[self.covariates].to_dict(),
        }
