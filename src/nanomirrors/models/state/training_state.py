from dataclasses import asdict, dataclass
from typing import Optional, List, Tuple, Dict
from dataclasses import field
from tqdm import tqdm
from pathlib import Path
import logging

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import anndata as ad
from huggingface_hub import hf_hub_download
from geomloss import SamplesLoss

from nanomirrors.data.perturbation_dataset import PairedSetPerturbationDataset
from nanomirrors.models.state.input_encoding_state import StateInputEncoderForPerturbation

from nanomirrors.evaluation.perturbation_eval import compute_perturbation_metrics


@dataclass
class StateTrainerConfig:
    lr: float = 1e-3
    max_steps: int = 20000
    batch_size: int = 32
    amp: bool = False # Use automatic mixed precision during training

    gradient_clip_norm: float = 10.0
    early_stopping_patience: int = 40 # Number of validation steps with no improvement after which training will be stopped

    val_every: int = 200
    log_every: int = 100

class StateTrainerForPerturbationResponsePrediction:

    def __init__(self, 
                 model: nn.Module,
                 train_data: ad.AnnData | None,
                 val_data: ad.AnnData | None,
                 test_data: ad.AnnData | None,
                 config: StateTrainerConfig,
                 device: torch.device,
                 perturbation_vocab: List[str],
                 set_covariates: Optional[List[str]] = None,
                 logger: Optional[logging.Logger] = None,
                 save_dir: str = "./"):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.config = config
        self.device = device
        self.perturbation_vocab = perturbation_vocab
        self.set_covariates = set_covariates
        self.logger = logger
        self.save_dir = save_dir

        self.cell_set_size = model.config.cell_set_size

        # Initialize the model-task specific dataset, dataloaders.
        if self.set_covariates is not None:
            for cov in self.set_covariates:
                if cov not in train_data.obs.columns:
                    raise ValueError(f"Covariate '{cov}' not found in training data. \
                                     Please ensure it is included in the AnnData object's obs.")

        if train_data is not None:
            self.train_dataset = PairedSetPerturbationDataset(train_data, 
                                                              set_size=self.cell_set_size,
                                                              include_ctrl_ctrl_pairs=True,
                                                              set_covariates=self.set_covariates)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)

        if val_data is not None:
            self.val_dataset = PairedSetPerturbationDataset(val_data, 
                                                            set_size=self.cell_set_size,
                                                            include_ctrl_ctrl_pairs=False,
                                                            set_covariates=self.set_covariates)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        if test_data is not None:
            self.test_dataset = PairedSetPerturbationDataset(test_data, 
                                                             set_size=self.cell_set_size,
                                                             include_ctrl_ctrl_pairs=False,
                                                             set_covariates=self.set_covariates)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False)


        # Initialize the input encoder.
        if self.model.config.use_pert_features:
            self.perturbation_feature_path = hf_hub_download(
                repo_id="windmirror/pretrained_embeddings",
                repo_type="dataset",
                filename="ESM2_pert_features.pt",
                revision="main",
            )
            self.perturbation_features = torch.load(
                self.perturbation_feature_path,
                map_location="cpu",
                weights_only=True,
            )
        
        self.train_input_encoder = StateInputEncoderForPerturbation(
            perturbation_vocab=self.perturbation_vocab,
            perturbation_features=self.perturbation_features,
            use_multi_hot=self.model.config.use_multi_hot,
            use_pert_features=self.model.config.use_pert_features,
        )
        self.eval_input_encoder = self.train_input_encoder

        self.loss_fn = SamplesLoss(loss="energy")
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.config.lr)
        self.amp = self.config.amp
        self.amp_dtype = (
            torch.bfloat16
            if self.device.type == "cpu"
            or (
                self.device.type == "cuda"
                and torch.cuda.is_bf16_supported()
            )
            else torch.float16
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                self.amp
                and self.device.type == "cuda"
                and self.amp_dtype == torch.float16
            ),
        ) # Only use scaler if AMP is enabled and using float16 on CUDA since bf16 does not require it

        self.global_step = 0

        print(f"Trainer initialized with device: {self.device}, amp: {self.amp}, amp_dtype: {self.amp_dtype}")

    def _log(self, message):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def loss_func(self, outputs, targets):
        loss = self.loss_fn(outputs, targets).mean(axis=0) # [B, S, D]

        return loss

    def train_step(self, batch, debug = False):
        self.model.train()
        self.optimizer.zero_grad()

        encoded_batch = self.train_input_encoder.encode(batch)
        encoded_batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in encoded_batch.items()
        }

        with torch.amp.autocast(
            device_type=self.device.type, 
            dtype=self.amp_dtype,
            enabled=self.amp):
            outputs = self.model(
                basal_exprs=encoded_batch["basal_exprs"],
                perts=encoded_batch["perts"],
                batch=encoded_batch.get("batch", None)
            )
            loss = self.loss_func(outputs, encoded_batch["targets"])

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip_norm,
            error_if_nonfinite=False if self.scaler.is_enabled() else True,
        )
        if not torch.isfinite(norm):
            self._log(
                f"Found infinite gradient. This may be caused by the gradient "
                f"scaler. The current scale is {self.scaler.get_scale()}. This warning "
                "can be ignored if no longer occurs after autoscaling of the scaler."
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        self.global_step += 1

        if debug:
            print(f"\nStep {self.global_step} | Loss: {loss.item():.4f} | Scaler: {self.scaler.get_scale()} | Norm: {norm:.4f}\n")

        return loss.detach().item()

    @torch.no_grad
    def val_step(self, batch, debug = False):
        self.model.eval()

        encoded_batch = self.eval_input_encoder.encode(batch)
        encoded_batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in encoded_batch.items()
        }

        with torch.amp.autocast(
            device_type=self.device.type, 
            dtype=self.amp_dtype,
            enabled=self.amp):
            outputs = self.model(
                basal_exprs=encoded_batch["basal_exprs"],
                perts=encoded_batch["perts"],
                batch=encoded_batch.get("batch", None)
            )
            loss = self.loss_func(outputs, encoded_batch["targets"])

        if debug:
            print(f"\nValidation | Loss: {loss.item():.4f}\n")

        return loss.detach().item()
    
    @torch.inference_mode()
    def test_step(self, batch, debug: bool = False) -> Tuple[float, torch.Tensor, torch.Tensor]:
        self.model.eval()
    
        encoded_batch = self.eval_input_encoder.encode(batch)
        encoded_batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in encoded_batch.items()
        }
        with torch.amp.autocast(
            device_type=self.device.type, 
            dtype=self.amp_dtype,
            enabled=self.amp):
            outputs = self.model(
                basal_exprs=encoded_batch["basal_exprs"],
                perts=encoded_batch["perts"],
                batch=encoded_batch.get("batch", None)
            )
            loss = self.loss_func(outputs, encoded_batch["targets"])
        
        if debug:
            print(f"\nTest | Loss: {loss.item():.4f}\n")
        
        return loss.detach().item(), outputs, encoded_batch["targets"]

    def train(self, checkpoint_path: Optional[str] = None, debug: bool = False):
        train_losses = []
        val_losses = []
        metrics_history = []

        self.best_val_metric = float("-inf")
        self.patience = 0

        while self.global_step < self.config.max_steps:   
            for batch in tqdm(self.train_loader, desc="Training"):
                loss = self.train_step(batch, debug=debug)
                train_losses.append({"step": self.global_step, "train_loss": loss})

                if self.global_step % self.config.log_every == 0 or self.global_step == 1:
                    self._log(f"Step {self.global_step}, Train Loss: {loss:.4f}")
    
                '''
                Perform validation every self.config.val_every steps.
                '''
                if self.global_step % self.config.val_every == 0 or self.global_step == 1:
                    if self.val_loader is None:
                        self._log("[Warning] Validation loader is not provided. No checkpoint will be saved.")
                        continue

                    val_loss, val_results, _, _, _, _ = self.evaluate(loader=self.val_loader, debug=debug)
                    val_losses.append({"step": self.global_step, "val_loss": val_loss})
                    metrics_history.append({"step": self.global_step, **val_results}) 

                    # Log the current step and validation results.
                    self._log(f"Step {self.global_step}")
                    for k, v in val_results.items():
                        self._log(f"{k}: {v}")

                    # Save best model based on the specified validation metric.
                    metric_category= "expression"
                    metric_name = "pearson_delta_all_genes"
                    if val_results[metric_category][metric_name] > self.best_val_metric:
                        self.best_val_metric = val_results[metric_category][metric_name]
                        self.save_checkpoint(checkpoint_path)
                        self._log(f"New best model saved with {metric_name}: {self.best_val_metric:.4f}")
                        self.patience = 0
                    else:
                        self.patience += 1
                        if self.patience >= self.config.early_stopping_patience:
                            self._log(f"Early stopping triggered after {self.global_step} steps.")
                            break
                            
                if self.global_step >= self.config.max_steps:
                    self._log(f"Maximum training steps reached: {self.global_step}, stopping training.")
                    break
        
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "metrics_history": metrics_history
        }
    
    def evaluate(self, loader, debug=False) -> tuple[float, dict, np.ndarray, np.ndarray, list, list]:
        n_genes = loader.dataset.adata.n_vars

        preds = [] #[n_cells, n_genes]
        gts = []
        perturbations = []
        covariates = []
        eval_losses = []
        
        bidx = 0
        for batch in tqdm(loader, desc="Evaluating"):
            loss, outputs, targets = self.test_step(batch, debug=debug)
            eval_losses.append(loss)

            # Flatten the outputs and targets to have shape (n_cells * cell_set_size, n_genes)
            preds.append(outputs.detach().float().view(-1, n_genes).cpu().numpy())
            gts.append(targets.detach().float().view(-1, n_genes).cpu().numpy())
            perturbations_extended = [[p] * self.cell_set_size for p in batch["perturbations"]]
            perturbations_extended = [p for sublist in perturbations_extended for p in sublist]
            perturbations.extend(perturbations_extended)
            covariates_extended = [[c] * self.cell_set_size for c in batch["covariates"]]
            covariates_extended = [c for sublist in covariates_extended for c in sublist]
            covariates.extend(covariates_extended)

            # bidx += 1

            # if bidx >= 1:
            #     break

        preds = np.concatenate(preds, axis=0)
        gts = np.concatenate(gts, axis=0)

        assert preds.shape == gts.shape, f"Predictions shape {preds.shape} does not match ground truth shape {gts.shape}"
        eval_loss = np.mean(eval_losses)
        print(f"Evaluation Loss: {eval_loss}")

        print("Evaluating perturbation response prediction metrics...")
        print(f"preds shape: {preds.shape}, gts shape: {gts.shape}") # (n_cells, cell_set_size,n_genes)
        ctrl_adata = loader.dataset.adata[loader.dataset.adata.obs['perturbation'] == 'control']
        eval_results = compute_perturbation_metrics(preds, 
                                                    gts, 
                                                    perturbations,
                                                    covariates,
                                                    ctrl_adata=ctrl_adata,
                                                    )
        return eval_loss, eval_results, preds, gts, perturbations, covariates

    def test(self, debug=False):
        return self.evaluate(self.test_loader, debug=debug)

    def save_checkpoint(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "model_config": asdict(self.model.config),
            "training_config": asdict(self.config),
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint["model_config"] != asdict(self.model.config):
            raise ValueError("Checkpoint model configuration does not match this model")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint["global_step"]
        self.config = checkpoint["training_config"]

    