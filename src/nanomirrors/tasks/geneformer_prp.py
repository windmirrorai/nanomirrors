import argparse
import json
import logging
import os
import random

import numpy as np
import torch
import anndata as ad
import perturb_datasets
from tqdm import tqdm
from torch.utils.data import DataLoader

from nanomirrors.models.geneformer.modeling_geneformer import GeneformerForPerturbationResponsePrediction
from nanomirrors.models.geneformer.configuration_geneformer import GeneformerConfig
from nanomirrors.models.geneformer.training_geneformer import GeneformerTrainerForPerturbationResponsePrediction, GeneformerTrainerConfig

MODEL_CHOICES = ["scGPT_human", 
                 "StateTransition", 
                 "Geneformer-V2-104M", 
                 "Geneformer-V1-10M"]
DATASETS_CHOICES = ["adamson2016"]

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

def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False, help="Path to the local input .h5ad file.")
    parser.add_argument("--input_url", type=str, required=False, help="URL to the input .h5ad file. Ignored if --input is provided.")
    parser.add_argument("--input_name", type=str, default="adamson2016", help="Name of the input dataset.", choices=DATASETS_CHOICES)
    parser.add_argument("--no_logging", action="store_true", help="Disable logging of the run.")
    parser.add_argument("--save_root", type=str, default="./prp_run", help="Directory to save logs and outputs.")

    parser.add_argument("--model", type=str, choices=MODEL_CHOICES, default="Geneformer-V2-104M", help="Type of model to train.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Mode to run the script in.")
    parser.add_argument("--eval_train", action="store_true", help="Evaluate the model on the training set.")
    
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
    Load the dataset and prepare train, validation, and test splits
    '''
    data = perturb_datasets.load_dataset(args.input_name)
    print(f"dataset loaded with shape {data.shape}")
    train_data, val_data, test_data = get_train_val_test_data(data)
    
    num_train_perturbations = len([x for x in train_data.obs['perturbation'].unique().tolist() if x != 'control'])
    num_val_perturbations = len([x for x in val_data.obs['perturbation'].unique().tolist() if x != 'control'])
    num_test_perturbations = len([x for x in test_data.obs['perturbation'].unique().tolist() if x != 'control'])

    '''
    Initialize the model and trainer
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    config = GeneformerConfig()
    model = GeneformerForPerturbationResponsePrediction(config, n_genes=train_data.shape[1])
    model = model.to(device)

    trainer_config = GeneformerTrainerConfig(batch_size=args.batch_size)
    trainer = GeneformerTrainerForPerturbationResponsePrediction(
        model=model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        config=trainer_config,
        device=device,
        logger=logger,
        save_dir=args.save_dir,
    )
    if args.mode == "train":
        history = trainer.train(checkpoint_path=f"{args.save_dir}/best_model.pt", debug=args.debug)

        # Save the training history.
        if args.save_dir:
            with open(f"{args.save_dir}/training_history.json", "w") as f:
                json.dump(history, f, indent=4)

    trainer.load_checkpoint(f"{args.save_dir}/best_model.pt")

    '''
    Evaluate the model on the training data
    '''
    if args.eval_train:
        train_loss, train_eval_results, train_preds, train_gts, train_perturbations, train_covariates = trainer.train_validate(debug=args.debug)

        # Save training evaluation results.
        np.savez(f"{args.save_dir}/train_eval_results.npz", 
                preds=train_preds, 
                gts=train_gts, 
                perturbations=train_perturbations, 
                covariates=train_covariates, 
                **train_eval_results)

        if args.save_dir:
            with open(f"{args.save_dir}/train_eval_metrics.json", "w") as f:
                json.dump({
                    "train_eval_results": train_eval_results,
                    "num_train_perturbations": num_train_perturbations,
                    "num_val_perturbations": num_val_perturbations,
                    "num_test_perturbations": num_test_perturbations,
                    }, f, indent=4)

        print(f"Training evaluation results saved to {args.save_dir}/train_eval_results.npz and {args.save_dir}/train_eval_metrics.json")

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



