import argparse
import os

import torch
import scanpy as sc

from nanomirrors.data import CellDataset
from nanomirrors.auto import AutoModel, AutoConfig, AutoInputEncoder

DEFAULT_INPUT_URL = "https://datasets.cellxgene.cziscience.com/d6761a21-e226-434f-9370-fbcc7e549aa0.h5ad"

MODEL_TYPE_CHOICES = ["scGPT", 
                      "StateTransition", 
                      "Geneformer-V2-104M", 
                      "Geneformer-V1-10M"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False, help="Path to the local input .h5ad file.")
    parser.add_argument("--input_url", type=str, required=False, default=DEFAULT_INPUT_URL, help="URL to the input .h5ad file. Ignored if --input is provided.")
    parser.add_argument("--output", type=str, default="scGPT_embeddings.npy", help="Path to save the output embeddings.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing the data.")

    parser.add_argument("--model_type", type=str, choices=MODEL_TYPE_CHOICES, default="Geneformer-V2-104M", help="Type of model to use for embedding.")

    args = parser.parse_args()

     # Load the input data.
    if args.input:
        adata = sc.read(args.input)
    else:
        print(f"Downloading input data from {args.input_url}...")
        os.makedirs("downloaded_data", exist_ok=True)
        if os.path.exists("downloaded_data/tmp.h5ad"):
            os.remove("downloaded_data/tmp.h5ad")
        adata = sc.read("downloaded_data/tmp.h5ad", backup_url=args.input_url)

    # !TODO:Ensure gene symbols, ensembl IDs are available in adata.var['gene_symbol'], adata.var['gene_ensembl'].
    print(f"Embedding adata of shape {adata.shape}...")

    dataset = CellDataset(adata)
    try:
        num_workers = min(len(os.sched_getaffinity(0)) - 1, args.batch_size)
    except AttributeError:
        num_workers = min(os.cpu_count(), 0, args.batch_size)
    dataloader = torch.utils.data.DataLoader(dataset,
                                  batch_size=args.batch_size, 
                                  shuffle=False,
                                  num_workers=num_workers, 
                                  pin_memory=True)

    batch = next(iter(dataloader))

    print(f"Loading model {args.model_type}...")
    model = AutoModel.from_pretrained(args.model_type)
    input_encoder = AutoInputEncoder.from_pretrained(args.model_type,
                                                     gene_names=adata.var['ensembl_id'].dropna().tolist())
    inputs = input_encoder.encode(batch)

    max_T = 120
    inputs["input_ids"] = inputs["input_ids"][:, :max_T]
    inputs["attention_mask"] = inputs["attention_mask"][:, :max_T]

    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: type {type(value)}, len {len(value) if hasattr(value, '__len__') else 'N/A'}")

    embeddings = model.encode(**inputs)
    print(f"Embeddings of shape {embeddings.shape}")