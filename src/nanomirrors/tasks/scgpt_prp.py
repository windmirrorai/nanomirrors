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

from nanomirrors.data.perturbation_dataset import PairedPerturbationDataset

from nanomirrors.models.scgpt.input_encoding_scgpt import scGPTInputEncoderForPerturbation
from nanomirrors.models.scgpt.modeling_scgpt import scGPTForPerturbationResponsePrediction
from nanomirrors.models.scgpt.configuration_scgpt import scGPTConfig
from nanomirrors.models.scgpt.training_scgpt import scGPTTrainerForPerturbationResponsePrediction, scGPTTrainerConfig

from nanomirrors.evaluation import compute_perturbation_metrics

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

def log(message, logger=None):
    if logger:
        logger.info(message)
    else:
        print(message)

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
    parser.add_argument("--input_name", type=str, help="Name of the input dataset.", choices=DATASETS_CHOICES)
    parser.add_argument("--no_logging", action="store_true", help="Disable logging of the run.")
    parser.add_argument("--save_root", type=str, default="./prp_run", help="Directory to save logs and outputs.")

    parser.add_argument("--model", type=str, choices=MODEL_CHOICES, default="scGPT_human", help="Type of model to train.")

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation.")
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
        log(f"Arguments: {args}", logger)

    '''
    Load the dataset and prepare train, validation, and test splits
    '''
    data = perturb_datasets.load_dataset(args.input_name)
    print(f"dataset loaded with shape {data.shape}")
    train_data, val_data, test_data = get_train_val_test_data(data)

    '''
    Initialize the model and trainer
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    config = scGPTConfig()
    model = scGPTForPerturbationResponsePrediction(config)
    model = model.to(device)

    trainer_config = scGPTTrainerConfig(batch_size=args.batch_size)
    trainer = scGPTTrainerForPerturbationResponsePrediction(
        model=model,
        train_data=train_data,
        val_data=val_data,
        config=trainer_config,
        device=device
    )
    history = trainer.train(checkpoint_path=f"{args.save_dir}/best_model.ckpt", debug=args.debug)

    trainer.load_checkpoint(f"{args.save_dir}/best_model.ckpt")

    '''
    Evaluate the model on the test data
    '''
    # [scGPT only] create a test set specific input encoder without truncation.
    test_input_encoder = scGPTInputEncoderForPerturbation(
        gene_names = test_data.var['gene_symbol'].tolist(),
        model_name = model.config.model_name,
        truncate_to_max_length=False,
    )

    test_dataset = PairedPerturbationDataset(test_data, include_ctrl_ctrl_pairs=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    preds = [] #[n_cells, n_genes]
    gts = []
    perturbations = []
    covariates = []
    test_losses = []
    
    bidx = 0
    for batch in tqdm(test_loader, desc="Testing"):
        loss, outputs, targets = trainer.test_step(batch, input_encoder=test_input_encoder, debug=True)
        test_losses.append(loss)

        preds.append(outputs.detach().cpu().numpy())
        gts.append(targets.detach().cpu().numpy())
        perturbations.extend(batch["perturbations"])
        covariates.extend(batch["covariates"])
        
        
        bidx += 1

        if bidx >= 1:
            break

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)

    assert preds.shape == gts.shape, f"Predictions shape {preds.shape} does not match ground truth shape {gts.shape}"
    test_loss = np.mean(test_losses)
    print(f"Test Loss: {test_loss}")

    print("Evaluating perturbation response prediction metrics...")
    print(f"preds shape: {preds.shape}, gts shape: {gts.shape}")
    eval_results = compute_perturbation_metrics(preds, 
                                                gts, 
                                                perturbations,
                                                covariates,
                                                ctrl_adata=test_data[test_data.obs['perturbation'] == 'control'],
                                                )
    for k, v in eval_results.items():
        print(f"{k}: {v}")
    
    # Save evaluation results.
    np.savez(f"{args.save_dir}/eval_results.npz", 
             preds=preds, 
             gts=gts, 
             perturbations=perturbations, 
             covariates=covariates, 
             **eval_results)
    if args.save_dir:
        with open(f"{args.save_dir}/eval_metrics.json", "w") as f:
            json.dump({"eval_results": eval_results}, f, indent=4)



