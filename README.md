<p align="center">
  <img src="./assets/nanomirrors.png" alt="nanomirrors" width="140"> 
</p>

<h3 align="center">nanomirrors</h3>
<h3 align="center">Minimal, readable, production-ready SOTA bio cell models</h3>

> [!NOTE]
> **🚧 nanomirrors is in beta, with the public launch coming soon.** APIs may change slightly and some features are still landing.
> Try it out and tell us what breaks: [open an issue](https://github.com/windmirrorai/nanomirrors/issues) and ⭐ the repo to follow along.

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

embeddings = model.encode_cell(**inputs)

```

### (coming soon) Train for perturbation response prediction

```python
config = StateConfig(input_dim=2000
                     output_dim=2000, 
                     pert_dim=2024)
model = StateModel(config)

trainer_config = StateTrainerConfig()
trainer = StateTrainerForPerturbationResponsePrediction(model, train_data, trainer_config)
trainer.train(checkpoint_path="./run")
```

## Supported models

| Model | Paper | Inference | Train | Verified vs. original | Weights license |
|---|---|:---:|:---:|---|---|
| **scGPT** | [Cui et al., 2024](https://www.nature.com/articles/s41592-024-02201-0) | ✅ | ✅ (PRP) | Bitwise equivalent, 1.38× faster | Inherits [scGPT license](https://github.com/bowang-lab/scGPT) |
| **Geneformer** | [Theodoris et al., 2023](https://www.nature.com/articles/s41586-023-06139-9) | ✅ | ✅ (PRP) | Bitwise equivalent | Inherits [Geneformer license](https://huggingface.co/ctheodoris/Geneformer) |
| **State** | [Adduri et al., 2025](https://www.cell.com/cell/fulltext/S0092-8674(26)00921-9) | ✅ | ✅ (PRP) | Bitwise equivalent | Inherits [State license](https://github.com/ArcInstitute/state) |

Want a model added? [Open an issue](../../issues).

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
2. Compare outputs on fixed inputs.

## Licensing

The nanomirrors **code** is released under the [Apache 2.0](./LICENSE).

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






