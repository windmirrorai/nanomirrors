import argparse
import os

import torch
import scanpy as sc

from nanomirrors.data import CellDataset
from nanomirrors.auto import AutoModel, AutoConfig, AutoInputEncoder
from nanomirrors.data.utils import convert_ensembl_to_symbol, convert_symbol_to_ensembl

DEFAULT_INPUT_URL = "https://datasets.cellxgene.cziscience.com/d6761a21-e226-434f-9370-fbcc7e549aa0.h5ad"
#DEFAULT_INPUT_URL = "https://drive.usercontent.google.com/download?id=10WC3WMmgZkZ_6LE9Eg7EOAOuC7q_1jeW&export=download&confirm=t"

MODEL_CHOICES = ["scGPT_human",
                 "StateTransition",
                 "Geneformer-V2-104M",
                 "Geneformer-V1-10M"]

# !TODO: move this inside auto.
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
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing the data.")

    parser.add_argument("--model", type=str, choices=MODEL_CHOICES, default="Geneformer-V2-104M", help="Type of model to use for embedding.")

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

    print(f"Making sure both gene symbols and ensembl ids exist ...")
    gene_ensembl_exists = False
    gene_symbol_exists = False
    if "gene_ensembl" in adata.var.columns:
        gene_ensembl_exists = True
    if "gene_symbol" in adata.var.columns:
        gene_symbol_exists = True

    if not gene_ensembl_exists and not gene_symbol_exists:
        adata.var_names_make_unique()
        if adata.var_names[0].startswith("ENS"):
            adata.var['gene_ensembl'] = adata.var_names
            gene_lists = adata.var["gene_ensembl"].tolist()
            ensembl_to_symbol = convert_ensembl_to_symbol(gene_lists, organism="homo_sapiens")
            adata.var["gene_symbol"] = adata.var["gene_ensembl"].map(ensembl_to_symbol)
        else:
            adata.var['gene_symbol'] = adata.var_names
            gene_lists = adata.var["gene_symbol"].tolist()
            symbol_to_ensembl = convert_symbol_to_ensembl(gene_lists, organism="homo_sapiens")
            adata.var["gene_ensembl"] = adata.var["gene_symbol"].map(symbol_to_ensembl)
        gene_ensembl_exists = True
        gene_symbol_exists = True
    elif gene_ensembl_exists and not gene_symbol_exists:
        gene_lists = adata.var["gene_ensembl"].tolist()
        ensembl_to_symbol = convert_ensembl_to_symbol(gene_lists, organism="homo_sapiens")
        adata.var["gene_symbol"] = adata.var["gene_ensembl"].map(ensembl_to_symbol)
        gene_symbol_exists = True
    elif gene_symbol_exists and not gene_ensembl_exists:
        gene_lists = adata.var["gene_symbol"].tolist()
        symbol_to_ensembl = convert_symbol_to_ensembl(gene_lists, organism="homo_sapiens")
        adata.var["gene_ensembl"] = adata.var["gene_symbol"].map(symbol_to_ensembl)
        gene_ensembl_exists = True
    assert gene_ensembl_exists and gene_symbol_exists, "Both gene_ensembl and gene_symbol must exist in adata.var"
    
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

    print(f"Loading model {args.model}...")
    # @TODO: need to pass both gene symbol, ensembl ids
    model = AutoModel.from_pretrained(args.model)
    gene_format = MODEL2GENE_FORMAT.get(args.model, "gene_symbol")
    input_encoder = AutoInputEncoder.from_pretrained(args.model,
                                                     gene_names=adata.var[gene_format].tolist())
    inputs = input_encoder.encode(batch)

    max_T = 120
    # inputs["input_ids"] = inputs["input_ids"][:, :max_T]
    # inputs["attention_mask"] = inputs["attention_mask"][:, :max_T]
    inputs["gene_tokens"] = inputs["gene_tokens"][:, :max_T]
    inputs["exprs"] = inputs["exprs"][:, :max_T]
    inputs["padding_mask"] = inputs["padding_mask"][:, :max_T]

    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: type {type(value)}, len {len(value) if hasattr(value, '__len__') else 'N/A'}")

    embeddings = model.encode_cell(**inputs)
    print(f"Embeddings of shape {embeddings.shape}")