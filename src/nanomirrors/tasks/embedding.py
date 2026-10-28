import argparse
import os

import torch
import numpy as np
import scanpy as sc
from tqdm import tqdm

from nanomirrors.data import CellDataset
from nanomirrors.auto import AutoModel, AutoInputEncoder
from nanomirrors.data.utils import add_gene_symbol_and_ensembl

DEFAULT_INPUT_URL = "https://datasets.cellxgene.cziscience.com/d6761a21-e226-434f-9370-fbcc7e549aa0.h5ad"

MODEL_CHOICES = ["scGPT_human",
                 "StateTransition",
                 "Geneformer-V2-104M",
                 "Geneformer-V1-10M"]

# TODO: maybe move this mapping.
MODEL2GENE_FORMAT = {
    "scGPT_human": "gene_symbol",
    "StateTransition": "gene_symbol",
    "Geneformer-V2-104M": "gene_ensembl",
    "Geneformer-V1-10M": "gene_ensembl"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False, help="Path to the local input .h5ad file.")
    parser.add_argument("--input_url", type=str, required=False, default=DEFAULT_INPUT_URL, help="URL to the input .h5ad file. Ignored if --input is provided.")
    parser.add_argument("--output", type=str, default="embeddings.npy", help="Path to save the output embeddings.")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing the data.")

    parser.add_argument("--model", type=str, choices=MODEL_CHOICES, default="scGPT_human", help="Type of model to use for embedding.")
    parser.add_argument("--mode", type=str, choices=["cell", "gene"], default="cell", help="Mode of embedding: 'cell' for cell embeddings, 'gene' for gene embeddings.")
    
    args = parser.parse_args()

    '''
    Load the input data.
    '''
    if args.input:
        adata = sc.read(args.input)
    else:
        print(f"Downloading input data from {args.input_url}...")
        os.makedirs("downloaded_data", exist_ok=True)
        if os.path.exists("downloaded_data/tmp.h5ad"):
            os.remove("downloaded_data/tmp.h5ad")
        adata = sc.read("downloaded_data/tmp.h5ad", backup_url=args.input_url)

    '''
    Ensure that both gene symbols and Ensembl IDs exist in the dataset.
    '''
    adata = add_gene_symbol_and_ensembl(adata, organism=adata.uns.get("organism", "human"))

    '''
    Dataset and dataloader preparation for embedding.
    '''
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
    '''
    Run model specific input encoding and embedding inference.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    model = AutoModel.from_pretrained(args.model)
    model.to(device)

    gene_format = MODEL2GENE_FORMAT.get(args.model, "gene_symbol")
    input_encoder = AutoInputEncoder.from_pretrained(args.model,
                                                     gene_names=adata.var[gene_format].tolist())


    print("Running embedding inference...")
    embeddings = []
    for i, batch in tqdm(enumerate(dataloader), desc="Running embedding inference", total=len(dataloader)):
        inputs = input_encoder.encode(batch)
        for k, v in inputs.items():
            if isinstance(v, np.ndarray):
                inputs[k] = torch.from_numpy(v).to(device)
            elif isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        if args.mode == "cell":
            embeds = model.encode_cell(**inputs)
        elif args.mode == "gene":
            embeds = model.encode_gene(**inputs)
        embeddings.append(embeds)

    embeddings = torch.cat(embeddings, dim=0)
    
    np.save(args.output, embeddings.cpu().numpy())
    print(f"Saved embeddings to {args.output}")

    if args.mode == "cell":
        print(f"Done. Cell embeddings of shape {embeddings.shape}")
    elif args.mode == "gene":
        print(f"Done. Gene embeddings of shape {embeddings.shape}")