import anndata as ad
from typing import Optional

PERTURBATION_METHODS = [
    "CRISPRi",
    "CRISPRa",
    "CRISPR-Cas9",
]

PERTURBATION_METHOD_KEY = "perturbation_method"
COVARIATES_KEY = "covariates"
ORGANISM_KEY = "organism"

PERTURBATION_COL = "perturbation"
IS_CTRL_COL = "is_control"
CTRL_TOKEN = "control"
COMBO_SEP = "+"

GENE_SYMBOLS_COL = "gene_symbol"
GENE_ENSEMBL_IDS_COL = "gene_ensembl"

def validate_adata(adata, type: Optional[str] = None):

    if type and type in ['perturbation']:
        required_obs = [
            PERTURBATION_COL,
            IS_CTRL_COL,
        ]
        required_uns = [
            PERTURBATION_METHOD_KEY,
            COVARIATES_KEY,
            ORGANISM_KEY,
        ]
    else:
        required_uns = [ORGANISM_KEY]
        required_obs = []

    required_vars = [
        GENE_SYMBOLS_COL,
        GENE_ENSEMBL_IDS_COL,
    ]

    required_layers = []

    for col in required_obs:
        if col not in adata.obs:
            raise ValueError(f"Missing obs column: {col}")

    for layer in required_layers:
        if layer not in adata.layers:
            raise ValueError(f"Missing adata.layers['{layer}']")

    for var in required_vars:
        if var not in adata.var:
            raise ValueError(f"Missing adata.var['{var}']")
    
    for uns in required_uns:
        if uns not in adata.uns:
            raise ValueError(f"Missing adata.uns['{uns}']")

    perts = adata.obs[PERTURBATION_COL].astype(str)

    if perts.isna().any():
        raise ValueError("Missing perturbation labels")

    if not perts.eq(CTRL_TOKEN).any():
        raise ValueError("No control perturbations found, or the control condition is not labeled as `control`. ")

    if not (adata.obs[IS_CTRL_COL] == perts.eq(CTRL_TOKEN)).all():
        raise ValueError("is_control inconsistent with perturbation")

    num_perts = perts.nunique()
    if num_perts == 0:
        raise ValueError(f"No unique perturbations found")
    if num_perts == 1:
        raise ValueError(f"Only one unique perturbation found")
    
    print(f"Data passed schema validation successfully.")

    return adata
