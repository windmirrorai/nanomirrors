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


class CellDataset(Dataset):
    """
    For iterating each cell in cellxgene anndata object.
    """
    def __init__(self, adata: ad.AnnData):
        self.adata = adata

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):

        return {
            "exprs": _to_numpy(self.adata.X[idx]).ravel(),
        }

    
