import scanpy as sc
from huggingface_hub import hf_hub_download

HF_REPO_ID = "windmirror/perturb_datasets"
HF_REVISION = "main"

_DATASETS = {
    "adamson2016": "adamson2016/processed.h5ad",
}


def load_dataset(
    name: str,
    revision: str = HF_REVISION,
    **kwargs,
) -> sc.AnnData:
    if name not in _DATASETS:
        raise ValueError(
            f"Unknown dataset {name!r}. "
            f"Available: {sorted(_DATASETS)}"
        )

    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename=_DATASETS[name],
        revision=revision,
    )

    return sc.read_h5ad(path, **kwargs)


def list_datasets():
    return sorted(_DATASETS)
