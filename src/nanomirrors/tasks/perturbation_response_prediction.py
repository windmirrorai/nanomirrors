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

from nanomirrors.data.perturbation_dataset import PairedPerturbationDataset, PairedSetPerturbationDataset
# from nanomirrors.models.state.input_encoding_state import StateInputEncoder
from torch.utils.data import DataLoader
# from nanomirrors.models.state.training_state import StateTrainerConfig, StateTrainerForPerturbationResponsePrediction

from nanomirrors.models.geneformer.input_encoding_geneformer import GeneformerInputEncoderForPerturbation
from nanomirrors.models.geneformer.modeling_geneformer import GeneformerForPerturbationResponsePrediction
from nanomirrors.models.geneformer.configuration_geneformer import GeneformerConfig
from nanomirrors.models.geneformer.training_geneformer import GeneformerTrainerForPerturbationResponsePrediction, GeneformerTrainerConfig

from nanomirrors.models.scgpt.input_encoding_scgpt import scGPTInputEncoderForPerturbation
from nanomirrors.models.scgpt.modeling_scgpt import scGPTForPerturbationResponsePrediction
from nanomirrors.models.scgpt.configuration_scgpt import scGPTConfig
from nanomirrors.models.scgpt.training_scgpt import scGPTTrainerForPerturbationResponsePrediction, scGPTTrainerConfig

from nanomirrors.evaluation import compute_perturbation_metrics

MODEL_CHOICES = ["scGPT_human", 
                 "StateTransition", 
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
    parser.add_argument("--output_dir", type=str, default="./prp_run", help="Path to save the output embeddings.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing the data.")
    parser.add_argument("--no_logging", action="store_true", help="Disable logging of the run.")
    parser.add_argument("--save_dir", type=str, default="./prp_run", help="Directory to save logs and outputs.")

    parser.add_argument("--model", type=str, choices=MODEL_CHOICES, default="Geneformer-V2-104M", help="Type of model to use for embedding.")

    args = parser.parse_args()

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    data = perturb_datasets.load_dataset('adamson2016')
    print(f"dataset loaded with shape {data.shape}")
    train_data, val_data, test_data = get_train_val_test_data(data)
    
    # config = StateConfig(input_dim=2000, output_dim=2000, pert_dim=2024)
    # model = StateModel(config) 
    # model = model.to(device)

    # trainer_config = StateTrainerConfig()
    # trainer = StateTrainerForPerturbationResponsePrediction(
    #     model=model,
    #     train_data=data,
    #     val_data=None,
    #     config=trainer_config,
    #     device=device
    # )
    # history = trainer.train(checkpoint_path="./ckpt/checkpoint.pth")

    config = GeneformerConfig.from_pretrained(name="geneformer-v2-104m")
    model = GeneformerForPerturbationResponsePrediction(config=config, n_genes=data.shape[1])

    model = model.to(device)

    trainer_config = GeneformerTrainerConfig(batch_size=2)
    trainer = GeneformerTrainerForPerturbationResponsePrediction(
        model=model,
        train_data=train_data,
        val_data=val_data,
        config=trainer_config,
        device=device
    )
    history = trainer.train(checkpoint_path="./ckpt/geneformer_checkpoint.pth")

    # config = scGPTConfig()
    # model = scGPTForPerturbationResponsePrediction(config)

    # model = model.to(device)

    # trainer_config = scGPTTrainerConfig(batch_size=8, max_steps=1)
    # trainer = scGPTTrainerForPerturbationResponsePrediction(
    #     model=model,
    #     train_data=train_data,
    #     val_data=val_data,
    #     config=trainer_config,
    #     device=device
    # )

    # trainer.load_checkpoint("./ckpt/scgpt_checkpoint.pth")




    # ===Evaluate the model on the test data====
    test_input_encoder = trainer.input_encoder
    # test_input_encoder = scGPTInputEncoderForPerturbation(
    #     gene_names = test_data.var['gene_symbol'].tolist(),
    #     model_name = model.config.model_name,
    #     truncate_to_max_length=False,
    # )

    test_dataset = PairedPerturbationDataset(test_data, include_ctrl_ctrl_pairs=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    preds = [] #[n_cells, n_genes]
    gts = []
    perturbations = []
    covariates = []
    test_losses = []
    
    bidx = 0
    for batch in tqdm(test_loader, desc="Testing"):
        loss, outputs, targets = trainer.test_step(batch, input_encoder=test_input_encoder, debug=True)
        test_losses.append(loss)

        import pdb; pdb.set_trace()

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
    
    # Save evaluation results
    np.savez("eval_results.npz", preds=preds, gts=gts, perturbations=perturbations, covariates=covariates, **eval_results)
    if args.save_dir:
        with open(f"{args.save_dir}/eval_metrics.json", "w") as f:
            json.dump({
                "eval_results": eval_results
            }, f, indent=4)



