<p align="center">
  <img src="./assets/nanomirrors.png" alt="nanomirrors" width="140"> 
</p>

<h3 align="center">nanomirrors</h3>
<h3 align="center">Minimal, readable, production-ready SOTA bio cell models</h3>

> Questions, ideas, or simply want to discuss more about AIxBio? [Join our Discord](https://discord.com/channels/1555659539536158822/1555659540794708051).
> Found a bug? [Open an issue](https://github.com/windmirrorai/nanomirrors/issues).


<!-- > [!NOTE]
> **🚧 nanomirrors is in beta, with the public launch coming soon.** APIs may change slightly and some features are still landing.
> Try it out and tell us what breaks: [open an issue](https://github.com/windmirrorai/nanomirrors/issues) and ⭐ the repo to follow along. -->

<!-- <p align="center">
  <a href="https://pypi.org/project/nanomirrors/"><img src="https://img.shields.io/pypi/v/nanomirrors.svg" alt="PyPI"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://huggingface.co/windmirror"><img src="https://img.shields.io/badge/🤗-weights-yellow.svg" alt="Hugging Face"></a>
  <!-- TODO: add Perturbation Arena / Twitter badges -->
</p>

---

**Nanomirrors** gives you clean PyTorch implementations of state-of-the-art bio cell models behind **one unified API and framework**. Run inference, fine-tune, and train in ~10 lines of code.

Nanomirrors reimplements each model from scratch. Every implementation is small enough to read in one sitting and verified numerically against the original.

## Why nanomirrors

- **One API, many models.** Swap `scGPT` for `Geneformer`, `State` and more (soon) by changing one string.
- **Readable.** Each model is a single, self-contained file of a few hundred lines. No need to jump between twenty open tabs.
- **Faithful.** Implementations are tested for numerical equivalence against the official code and in reference to the paper.
- **Iterate quickly.** Want to test a new architecture, objective, or input encoding?  Change a few lines and find out.

## Install

```bash
git clone https://github.com/windmirrorai/nanomirrors
cd nanomirrors
```
Option A: using uv
```bash
uv sync
source .venv/bin/activate
```
Option B: using pip
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Tutorials

Step-by-step tutorials to help you get started:

| Tutorial | Example Code | Colab Notebook |
|---|---|---|
| Cell embeddings | [View](./example.py) | [Open in Colab](https://colab.research.google.com/drive/1t5_HpgOD0_60v5qSE9wgGPK-WzaYwkSK?usp=sharing) |
| Perturbation response prediction | [View](./example_perturbation_response_prediction.py) | - |
<!-- | Preparing custom data | [View](tutorials/03_custom_data.ipynb) | [Open in Colab](https://colab.research.google.com/github/windmirrorai/nanomirrors/blob/main/tutorials/03_custom_data.ipynb) | -->

## Quick Start

```python
# scGPT cell embedding  example
import numpy as np
from nanomirrors.auto import AutoModel, AutoInputEncoder

genes = ['DUX4L30', 'USP17L16P', 'RPL7P23', 'ZZZ3', 'ZSCAN9', 'VN1R104P']
exprs = np.array([[1.0, 0.0, 23.0, 6.0, 5.0, 4.0], [0.0, 0.0, 3.0, 7.0, 2.0, 1.0]])

model_name = "scGPT_human" # or "Geneformer-V2-104M", "Geneformer-V1-10M"

model = AutoModel.from_pretrained(model_name)
model.eval()

input_encoder = AutoInputEncoder.from_pretrained(model_name, gene_names=genes)
inputs = input_encoder.encode({"exprs": exprs})

cell_embeddings = model.encode_cell(**inputs) # (N, D)
gene_embeddings = model.encode_gene(**inputs) # (N, G, D)

```

### or using CLI

```bash
# Example: Tabula Sapiens lung data (downloaded automatically)
python tasks/embedding.py \
    --model scGPT_human \
    --output embeddings.npy

# Use a local .h5ad file
python tasks/embedding.py \
    --model Geneformer-V2-104M \
    --input <path to local .h5ad file> \
    --output <path to save embeddings>

# Download a remote .h5ad file
python tasks/embedding.py \
    --model scGPT_human \
    --input_url <URL to a remote .h5ad file> \
    --output <path to save embeddings>
```

Supported embedding models: `scGPT_human`, `Geneformer-V2-104M`, and `Geneformer-V1-10M`.

## Perturbation Response Prediction (PRP) task
Using your own dataset? See [Bring your own data](#bring-your-own-data) for the required format and annotations.

```python
import torch
from nanomirrors.data import load_dataset
from nanomirrors.auto import AutoTrainerForPerturbationResponsePrediction

data = load_dataset("adamson2016") # load preprocessed perturbation dataset from HF

# Automatically intialize model, configs, and corresponding trainer 
trainer = AutoTrainerForPerturbationResponsePrediction.from_model(
    model_name="State",
    n_genes=data.shape[1], # number of genes in data
    device=torch.device("cuda")
)

history = trainer.train(train_data=data, val_data=None, checkpoint_path="./run")

```

or use CLI

```sh
cd /src/nanomirrors

# Example: train/evaluate on adamson2016 dataset (preprocessed, downloaded automatically)
python tasks/perturbation_response_prediction.py \
    --input_name adamson2016 \
    --model State
    --mode <train/test>

# Or on your custom dataset
python tasks/finetune_perturbation.py  \
    --input <path to local .h5ad file> \
    --model scGPT
    --mode <train/test>

# Or data from remote URL
python tasks/finetune_perturbation.py  \
    --input_url <path to local .h5ad file> \
    --model Geneformer-V2-104M
    --mode <train/test>

```
Supported PRP models: `State`, `scGPT_human`, `Geneformer-V2-104M`, and `Geneformer-V1-10M`.

## Bring your own data

Use an `AnnData` object with **cells as rows and genes as columns**, plus these annotations:

| Field | Contents |
|---|---|
| `adata.var["gene_symbol"]` | Gene symbols, e.g. `"TP53"` |
| `adata.var["gene_ensembl"]` | Ensembl gene IDs, e.g. `"ENSG00000141510"` |
| `adata.uns["organism"]` | Organism, e.g. `"human"` |

For **perturbation response prediction**, also include:

| Field | Contents |
|---|---|
| `adata.obs["perturbation"]` | Target gene, `"+"`-joined targets such as `"TP53+MDM2"`, or `"control"` |
| `adata.obs["is_control"]` | Boolean indicating whether each cell is a control |
| `adata.uns["perturbation_method"]` | `"CRISPRi"`, `"CRISPRa"`, or `"CRISPR-Cas9"` |
| `adata.uns["covariates"]` | List of covariate column names in `.obs`, e.g. `"[cell_type]"`or `[]` if no cell covariates. Covariates influences cells grouping in training. |

```python
# Assuming adata.obs["perturbation"], adata.var["gene_ensembl"] and adata.var["gene_symbols"] already populated:
adata.uns["organism"] = "human"
adata.uns["perturbation_method"] = "CRISPRi"
adata.uns["covariates"] = ["cell_type"]  # Requires adata.obs["cell_type"]

adata.obs["is_control"] = adata.obs["perturbation"].eq("control")
```

Include control cells and at least one non-control perturbation, with no missing perturbation labels. Example: [preprocessed adamson perturbation h5ad](https://huggingface.co/datasets/windmirror/perturb_datasets/blob/main/adamson2016/processed.h5ad). 

## Supported models and tasks

Use the model identifier below when loading a model or trainer.

| Model | Model identifier | Embedding | PRP |
|---|---|:---:|:---:|
| **scGPT** | `"scGPT_human"` | ✅ | ✅ |
| **Geneformer V2** | `"Geneformer-V2-104M"` | ✅ | ✅ |
| **Geneformer V1** | `"Geneformer-V1-10M"` | ✅ | ✅ |
| **State** | `"State"` | — | (ST+HVG) ✅ , (SE+ST) coming soon  |

**PRP** = perturbation response prediction. 

### References and verification

| Model | Paper | Verified vs. original | Weights license |
|---|---|---|---|
| **scGPT** | [Cui et al., 2024](https://www.nature.com/articles/s41592-024-02201-0) | Equivalent; 1.38× faster in our benchmark | [scGPT license](https://github.com/bowang-lab/scGPT) |
| **Geneformer** | [Theodoris et al., 2023](https://www.nature.com/articles/s41586-023-06139-9) | Equivalent | [Geneformer license](https://huggingface.co/ctheodoris/Geneformer) |
| **State** | [State paper](https://www.cell.com/cell/fulltext/S0092-8674%2826%2900921-9) | Equivalent | [State license](https://github.com/ArcInstitute/state) |

#### Want another model or task supported? [Open an issue](https://github.com/windmirrorai/nanomirrors/issues).

<!-- ## Design

Each model is assembled from three swappable pieces:

```
Tokenizer ──► InputEncoder ──► Model
(genes → ids)  (expression → tensors)  (transformer + heads)
```

For example, `GeneformerTokenizer` / `GeneformerInputEncoder` mirror `ScGPTTokenizer` / `ScGPTInputEncoder`. A new model needs only these three components plus a weight-loading function. See [`CONTRIBUTING.md`](./CONTRIBUTING.md).

```
nanomirrors/
├── models/
│   ├── scgpt.py
│   ├── geneformer.py
│   └── state.py
├── tokenizers/
├── tasks/          # fine-tuning pipelines (perturbation, annotation, …)
└── hub.py          # Hugging Face weight loading
``` -->

## Numerical Equivalence

Reimplementations are only useful if you can trust them. For every model we:

1. Load the official checkpoint into both the reference code and the nanomirrors version.
2. Compare outputs on fixed inputs. Equivalence is checked with `torch.allclose(ours, ref, atol=1e-3)`

## Licensing

The nanomirrors code is released under the [Apache 2.0](./LICENSE).

Each wrapped model's **pretrained weights and model-specific terms** inherit the license of the original model. Check the table above before using a model commercially.

## Citation

```bibtex
@software{nanomirrors2026,
  author = {Danqi Liao, Ann Yu},
  title  = {nanomirrors: Minimal, readable state-of-the-art cell models},
  year   = {2026},
  url    = {https://github.com/windmirrorai/nanomirrors}
}
```

Please also cite the original paper for each model you use.

---






