import argparse
import json
import logging
import os

import numpy as np
import torch
import anndata as ad
import perturb_datasets
from tqdm import tqdm

from typing import Dict, List, Tuple
import scanpy as sc
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist

from nanomirrors.auto import AutoModel, AutoInputEncoder

MODEL_CHOICES = ["scGPT_human", 
                 "State", 
                 "Geneformer-V2-104M", 
                 "Geneformer-V1-10M"]

PERTURBATION_METHODS = [
    "CRISPRi",
    "CRISPRa",
    "CRISPR-Cas9",
]

def get_train_val_test_data(adata: ad.AnnData, split_col: str = "split"):
    train_data = adata[(adata.obs[split_col] == "train") | (adata.obs[split_col] == "control")]
    val_data = adata[(adata.obs[split_col] == "val") | (adata.obs[split_col] == "control")]
    test_data = adata[(adata.obs[split_col] == "test") | (adata.obs[split_col] == "control")]

    return train_data, val_data, test_data

def log(message, logger=None):
    if logger:
        logger.info(message)
    else:
        print(message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False, help="Path to the local input .h5ad file.")
    parser.add_argument("--input_url", type=str, required=False, help="URL to the input .h5ad file. Ignored if --input is provided.")
    parser.add_argument("--input_name", type=str, default="adamson2016", help="Name of the input dataset.", choices=DATASETS_CHOICES)

    parser.add_argument("--model", type=str, choices=MODEL_CHOICES, default="State", help="Type of model to train.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation.")

    parser.add_argument("--cell_covariates", type=list, default=[], help="list of cell covariates, e.g., ['batch', 'cell_type']")
    parser.add_argument("--cell_set_size", type=int, default=64, help="Cell set size for the model.")
    parser.add_argument("--use_multi_hot", action="store_true", help="Use multi-hot encoding for perturbations.")
    parser.add_argument("--use_pert_features", action="store_true", help="Use perturbation features for encoding.")

    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Mode to run the script in.")
    
    parser.add_argument("--no_logging", action="store_true", help="Disable logging of the run.")
    parser.add_argument("--save_root", type=str, default="./prp_run", help="Directory to save logs and outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")    
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
        
    args = parser.parse_args()

    if args.model in ['state']:
        if not args.use_multi_hot and not args.use_pert_features:
            raise ValueError("At least one of --use_multi_hot or --use_pert_features must be specified.")
        if args.use_multi_hot and args.use_pert_features:
            print(f"[WARNING] Both --use_multi_hot and --use_pert_features were specified. Disabling --use_pert_features.")
            args.use_pert_features = False
    

    seed_everything(args.seed)

    '''
    Setup logging and output directories
    '''
    if args.use_pert_features:
        args.save_dir = f"{args.save_root}/{args.input_name}/{args.model}-pert_features"
    else:
        args.save_dir = f"{args.save_root}/{args.input_name}/{args.model}-multi_hot"
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
    Load the dataset and prepare train, validation, and test splits
    '''
    data = perturb_datasets.load_dataset(args.input_name)
    print(f"dataset loaded with shape {data.shape}")
    train_data, val_data, test_data = get_train_val_test_data(data)
    
    num_train_perturbations = len([x for x in train_data.obs['perturbation'].unique().tolist() if x != 'control'])
    num_val_perturbations = len([x for x in val_data.obs['perturbation'].unique().tolist() if x != 'control'])
    num_test_perturbations = len([x for x in test_data.obs['perturbation'].unique().tolist() if x != 'control'])
    total_perturbations = num_train_perturbations + num_val_perturbations + num_test_perturbations
    n_genes = train_data.shape[1]
    
    '''
    Initialize the model and trainer
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    model = AutoModel.from_pretrained(args.model)
    model = model.to(device)

    # TODO: pass set_covariates from dataset
    perturbation_vocab = data.obs['perturbation'].unique().tolist()
    trainer = AutoTrainerForPerturbationResponsePrediction(
        model=model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        batch_size=args.batch_size,
        device=device,
        perturbation_vocab=perturbation_vocab,
        covariates=args.cell_covariates,
        logger=logger,
        save_dir=args.save_dir,
    )
    # trainer_config = StateTrainerConfig(batch_size=args.batch_size)
    # trainer = StateTrainerForPerturbationResponsePrediction(
    #     model=model,
    #     train_data=train_data,
    #     val_data=val_data,
    #     test_data=test_data,
    #     config=trainer_config,
    #     device=device,
    #     perturbation_vocab=data.obs['perturbation'].unique().tolist(),
    #     set_covariates=None,
    #     logger=logger,
    #     save_dir=args.save_dir,
    # )
    if args.mode == "train":
        history = trainer.train(checkpoint_path=f"{args.save_dir}/best_model.pt", debug=args.debug)

        # Save the training history.
        if args.save_dir:
            with open(f"{args.save_dir}/training_history.json", "w") as f:
                json.dump(history, f, indent=4)

    trainer.load_checkpoint(f"{args.save_dir}/best_model.pt")

    '''
    Evaluate the model on the test data
    '''
    test_loss, test_eval_results, preds, gts, perturbations, covariates = trainer.test(debug=args.debug)

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











