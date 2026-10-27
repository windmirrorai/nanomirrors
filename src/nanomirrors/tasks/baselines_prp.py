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
from nanomirrors.evaluation import compute_perturbation_metrics

MODEL_CHOICES = ["train-mean", "no-change", "split-half"]
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

    parser.add_argument("--model", type=str, choices=MODEL_CHOICES, default="train-mean", help="Type of baseline model to use.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation.")

    parser.add_argument("--cell_covariates", default=[], type=list, help="The covariates for the cells. e.g ['batch', 'cell_type']")
    
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
    total_perturbations = num_train_perturbations + num_val_perturbations + num_test_perturbations
    n_genes = train_data.shape[1]
    
    '''
    Initialize the model and trainer
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    
    # Initialize the dataset and dataloaders.
    train_dataset = PairedPerturbationDataset(train_data, pair_covariates=args.cell_covariates, include_ctrl_ctrl_pairs=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    train_mean = np.zeros(n_genes) # global mean vector across all training conditions
    train_means = {} # dict[condition, mean_vector]; condition key is "perturbation:<perturbation>_<covariate_key>:<covariate_value>..."
    if args.model in ["train-mean"]:
        idx = 0

        for batch in tqdm(train_loader, desc="training"):
            exprs = batch['exprs'].squeeze(axis=1) # (batch_size, 1, n_genes) -> (batch_size, n_genes)
            perts = batch['perturbations']
            covariates = batch['covariates']
            
            if covariates and len(covariates) > 0:
                for pert, expr, cov in zip(perts, exprs, covariates):
                    cond_key = f"perturbation:{pert}"
                    cond_key += "_".join([f"{k}:{v}" for k, v in cov.items()])
                    if cond_key not in train_means:
                        train_means[cond_key] = []
                    train_means[cond_key].append(expr.cpu().float().numpy())
            else:
                for pert, expr in zip(perts, exprs):
                    cond_key = f"perturbation:{pert}"
                    if cond_key not in train_means:
                        train_means[cond_key] = []
                    train_means[cond_key].append(expr.cpu().float().numpy())

            # idx += 1
            # if idx > 5:
            #     break

        # Compute the mean vectors for each condition.
        for cond_key in train_means:
            train_means[cond_key] = np.mean(train_means[cond_key], axis=0)
        print(f"Computed train means for {len(train_means)} conditions.")
        for cond_key, mean_vector in train_means.items():
            print(f"Condition: {cond_key}, Mean vector shape: {mean_vector.shape}")
        train_mean = np.mean(list(train_means.values()), axis=0)

        # save the computed train means to a file
        save_path = os.path.join(args.output_dir, "train_means.npy")
        np.save(save_path, train_means)
        print(f"Saved train means to {save_path}")


    '''
    Evaluate the model on the test data
    '''
    test_dataset = PairedPerturbationDataset(test_data, pair_covariates=args.cell_covariates, include_ctrl_ctrl_pairs=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    gts = [] # [n_cells, n_genes]
    preds = []
    perturbations = []
    covariates = []
    idx = 0
    for batch in tqdm(test_loader, desc="evaluating"):

        batch_preds = None
        if args.model == "train-mean":
            batch_preds = np.tile(train_mean, (batch['exprs'].shape[0], 1))
        elif args.model in ['no-change', 'split-half']:
            batch_preds = batch['ctrl_exprs'].squeeze(axis=1).cpu().float().detach().numpy()
        else:
            raise NotImplementedError(f"Model {args.model} is not implemented yet.")
    
        preds.append(batch_preds)
        gts.append(batch['exprs'].squeeze(axis=1).cpu().float().detach().numpy())
        perturbations.extend(batch['perturbations'])
        covariates.extend(batch['covariates'])

        # idx += 1
        # if idx > 5:
        #     break
    
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)

    if args.model == "split-half":
        condition2idx = {}
        if covariates and len(covariates) > 0:
            for i, (pert, covs) in enumerate(zip(perturbations, covariates)):
                key = f"perturbation:{pert}"
                for cov_name, cov_value in covs.items():
                    key += f"_{cov_name}:{cov_value}"
                if key not in condition2idx:
                    condition2idx[key] = []
                condition2idx[key].append(i)
        else:
            for i, pert in enumerate(perturbations):
                key = f"perturbation:{pert}"
                if key not in condition2idx:
                    condition2idx[key] = []
                condition2idx[key].append(i)
   
        # Split each condition's indices into two halves: one for ground truth and one for predictions.
        rng = np.random.default_rng(args.seed)
        gts_idx, preds_idx = [], []
        for key, idx_list in condition2idx.items():
            idx_list = rng.permutation(idx_list).tolist()
            half = len(idx_list) // 2
            gts_idx.extend(idx_list[:half])
            preds_idx.extend(idx_list[half:half*2]) # omit the remaining index if the list length is odd.
        
        import pdb; pdb.set_trace()
        preds = gts[preds_idx].copy()
        gts = gts[gts_idx].copy()
        perturbations = [perturbations[i] for i in gts_idx]
        if covariates and len(covariates) > 0:
            covariates = [covariates[i] for i in gts_idx]

        print(f"Split-half selection: {len(gts_idx)} ground truth indices, {len(preds_idx)} prediction indices.")


    assert preds.shape == gts.shape, f"Predictions shape {preds.shape} does not match ground truth shape {gts.shape}"
    print("Evaluating perturbation response prediction metrics...")
    print(f"preds shape: {preds.shape}, gts shape: {gts.shape}")
    ctrl_adata = test_dataset.adata[test_dataset.adata.obs['perturbation'] == 'control']
    test_eval_results = compute_perturbation_metrics(preds, 
                                                    gts, 
                                                    perturbations,
                                                    covariates,
                                                    ctrl_adata=ctrl_adata,
                                                    )

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



