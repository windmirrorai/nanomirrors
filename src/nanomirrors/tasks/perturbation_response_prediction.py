import argparse
import json
import random
import logging
import os
from tqdm import tqdm

from typing import Dict, List, Tuple
from dataclasses import asdict

import numpy as np
import torch
import anndata as ad
import scanpy as sc

from nanomirrors.data import load_dataset, list_datasets, validate_adata
from nanomirrors.auto import list_models
from nanomirrors.auto import AutoTrainerForPerturbationResponsePrediction

MODEL_CHOICES = ["scGPT_human", 
                 "State", 
                 "Geneformer-V2-104M", 
                 "Geneformer-V1-10M"]
DATASETS_CHOICES = ["adamson2016"]

def get_train_val_test_data(adata: ad.AnnData, split_col: str = "split"):
    if split_col not in adata.obs:
        print(f"Split column '{split_col}' not found in adata.obs. Randomly assigning cells to splits.")
        num_cells = adata.n_obs
        pert_idx = np.flatnonzero(~adata.obs["is_control"])
        split_values = np.array(["control"] * num_cells)
        split_values[pert_idx] = np.random.choice(["train", "val", "test"], size=len(pert_idx), p=[0.7, 0.1, 0.2])
        adata.obs[split_col] = split_values
    else:
        if not set(adata.obs[split_col].unique()).issuperset({"train", "val", "test", "control"}):
            raise ValueError(f"Split column '{split_col}' must contain at least 'train', 'val', 'test', and 'control' values.")
        print(f"Using existing split column '{split_col}' with values: {adata.obs[split_col].unique()}")

    train_data = adata[(adata.obs[split_col] == "train") | (adata.obs[split_col] == "control")]
    val_data = adata[(adata.obs[split_col] == "val") | (adata.obs[split_col] == "control")]
    test_data = adata[(adata.obs[split_col] == "test") | (adata.obs[split_col] == "control")]

    return train_data, val_data, test_data

def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log(message, logger=None):
    if logger:
        logger.info(message)
    else:
        print(message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False, help="Path to the local input .h5ad file.")
    parser.add_argument("--input_url", type=str, required=False, help="URL to the input .h5ad file. Ignored if --input is provided.")
    parser.add_argument("--input_name", type=str, default="adamson2016", help="Name of the input dataset.", choices=list_datasets())
    parser.add_argument("--split_col", type=str, default="split", help="Column in adata.obs to use for train/val/test splits.")

    parser.add_argument("--model", type=str, choices=list_models(), default="State", help="Type of model to train.")
    parser.add_argument("--batch_size", type=int, help="Batch size for training and evaluation.")

    parser.add_argument("--cell_set_size", type=int, help="Cell set size for the model.")
    parser.add_argument("--use_multi_hot", action="store_true", help="Use multi-hot encoding for perturbations.")

    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Mode to run the script in.")
    
    parser.add_argument("--no_logging", action="store_true", help="Disable logging of the run.")
    parser.add_argument("--save_root", type=str, default="./prp_run", help="Directory to save logs and outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")    
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
        
    args = parser.parse_args()

    seed_everything(args.seed)

    '''
    Setup logging and output directories
    '''
    args.save_dir = f"{args.save_root}/{args.input_name}/{args.model}"
    os.makedirs(args.save_dir, exist_ok=True)

    logger = None
    if not args.no_logging:
        logging.basicConfig(
            filename=f"{args.save_dir}/run.log",
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger(__name__)
    if logger:
        logger.info(f"Arguments: {args}")
    else:
        print(f"Arguments: {args}")

    '''
    Load the dataset and prepare train, validation, and test splits.
    '''
    if args.input is not None:
        data = sc.read(args.input)
    elif args.input_url is not None:
        print(f"Downloading input data from {args.input_url}...")
        os.makedirs("downloaded_data", exist_ok=True)
        if os.path.exists("downloaded_data/tmp.h5ad"):
            os.remove("downloaded_data/tmp.h5ad")
        data = sc.read("downloaded_data/tmp.h5ad", backup_url=args.input_url)
    else:
        data = load_dataset(args.input_name)

    validate_adata(data, type="perturbation")
    print(f"dataset loaded with shape {data.shape}")

    # NOTE: `split_col` in `adata.obs` will be used to determine the splits. 
    # `split_col` must contain at least the values `train`, `val`, `test`, and `control`.
    # If it does not exist, random splits will be assigned to the cells, with control cells labeled as `control`.
    train_data, val_data, test_data = get_train_val_test_data(data, split_col=args.split_col)
    
    num_train_perturbations = len([x for x in train_data.obs['perturbation'].unique().tolist() if x != 'control'])
    num_val_perturbations = len([x for x in val_data.obs['perturbation'].unique().tolist() if x != 'control'])
    num_test_perturbations = len([x for x in test_data.obs['perturbation'].unique().tolist() if x != 'control'])
    
    n_genes = train_data.shape[1]
    
    '''
    Initialize the model, trainer config and trainer
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    # Override default trainer config, model config if supplied via args.
    model_config_params = {}
    trainer_config_params = {}
    if args.cell_set_size is not None:
        model_config_params = {"cell_set_size": args.cell_set_size}
    if args.batch_size is not None:
        trainer_config_params = {"batch_size": args.batch_size}
        
    trainer = AutoTrainerForPerturbationResponsePrediction.from_model(
        model_name=args.model,
        n_genes=n_genes,
        trainer_config_params=trainer_config_params,
        model_config_params=model_config_params,
        device=device,
        logger=logger,
        save_dir=args.save_dir,
    )

    print("Trainer Config: ", asdict(trainer.config))
    print("Model Config: ", asdict((trainer.model.config)))
    print("Device: ", device)

    if args.mode == "train":
        history = trainer.train(train_data=train_data, 
                                val_data=val_data, 
                                checkpoint_path=f"{args.save_dir}/best_model.pt", 
                                debug=args.debug)

        # Save the training history.
        if args.save_dir:
            with open(f"{args.save_dir}/training_history.json", "w") as f:
                json.dump(history, f, indent=4)

    trainer.load_checkpoint(f"{args.save_dir}/best_model.pt")

    '''
    Evaluate the model on the test data
    '''
    test_loss, test_eval_results, preds, gts, perturbations, covariates = trainer.test(data=test_data, debug=args.debug)

    # Save evaluation results.
    np.savez(f"{args.save_dir}/eval_results.npz", 
             preds=preds, 
             gts=gts, 
             perturbations=perturbations, 
             covariates=covariates, 
             **test_eval_results)
    
    if args.save_dir:
        with open(f"{args.save_dir}/eval_metrics.json", "w") as f:
            json.dump({
                "eval_results": test_eval_results,
                "num_train_perturbations": num_train_perturbations,
                "num_val_perturbations": num_val_perturbations,
                "num_test_perturbations": num_test_perturbations,
                }, f, indent=4)
    
    print(f"Done. Saved results to {args.save_dir} .")
