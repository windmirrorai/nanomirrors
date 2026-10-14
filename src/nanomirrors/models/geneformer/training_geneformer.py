from dataclasses import asdict, dataclass
from typing import Optional, Tuple
from dataclasses import field
from tqdm import tqdm
from pathlib import Path
import logging

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import anndata as ad

from nanomirrors.data.perturbation_dataset import PairedPerturbationDataset
from nanomirrors.models.geneformer.input_encoding_geneformer import GeneformerInputEncoderForPerturbation

from nanomirrors.evaluation import compute_perturbation_metrics

@dataclass
class GeneformerTrainerConfig:
    lr: float = 1e-3
    max_steps: int = 80000
    batch_size: int = 32
    amp: bool = True # Use automatic mixed precision during training
    scheduler_gamma: float = 0.9

    early_stopping_patience: int = 20 # Number of validation steps to wait before early stopping
    gradient_clip_norm: float = 10.0

    val_every: int = 800
    log_every: int = 100

class GeneformerTrainerForPerturbationResponsePrediction:

    def __init__(self, 
                 model: nn.Module,
                 train_data: ad.AnnData,
                 val_data: ad.AnnData | None,
                 test_data: ad.AnnData | None,
                 config: GeneformerTrainerConfig,
                 device: torch.device,
                 pair_covariates: Optional[torch.Tensor] = None,
                 logger : Optional[logging.Logger] = None,
                 save_dir: str = "./"):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.config = config
        self.device = device
        self.pair_covariates = pair_covariates
        self.logger = logger
        self.save_dir = save_dir
        
        # Initialize the model-task specific dataset, dataloaders.
        if train_data is not None:
            self.train_dataset = PairedPerturbationDataset(train_data, pair_covariates=pair_covariates)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        if val_data is not None:
            self.val_dataset = PairedPerturbationDataset(val_data, pair_covariates=pair_covariates, include_ctrl_ctrl_pairs=False)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        if test_data is not None:
            self.test_dataset = PairedPerturbationDataset(test_data, pair_covariates=pair_covariates, include_ctrl_ctrl_pairs=False)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Initialize the input encoder.
        perturbation_vocab = self.train_dataset.perturbations
        if "v2" in model.config.model_name:
             tokenizer_name = "geneformer_V2"
        elif "v1" in model.config.model_name:
             tokenizer_name = "geneformer_V1"
        else:
             raise ValueError(f"Unknown model name: {model.config.model_name}")
        self.train_input_encoder = GeneformerInputEncoderForPerturbation(
            gene_names = train_data.var['gene_ensembl'].dropna().tolist(),
            perturbation_method = train_data.uns['meta']['dataset']['perturbation_method'],
            model_name = tokenizer_name
        )
        self.eval_input_encoder = self.train_input_encoder # same input encoder for Geneformer model.
        
        self.loss_fn = nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                                  lr=self.config.lr)
        scheduler_step_size = len(self.train_loader) # step decay LR per one epoch.
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=config.scheduler_gamma)
        
        self.amp = config.amp
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
        loss = self.loss_fn(outputs, targets) # [B, T] -> [B] -> scalar

        return loss
    
    def train_step(self, batch, debug: bool = False):
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
                ctrl_input_ids = encoded_batch["ctrl_input_ids"],
                ctrl_attention_mask = encoded_batch["ctrl_attention_mask"],
                edited_input_ids = encoded_batch["edited_input_ids"],
                edited_attention_mask = encoded_batch["edited_attention_mask"]
            ) # [B, T]

            loss = self.loss_func(outputs, encoded_batch["targets"])

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer) # has to explicitly unscale before clipping gradients.
        
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

        self.scheduler.step()

        if debug:
            print(f"\nStep {self.global_step} | Loss: {loss.item():.4f} | LR: {self.scheduler.get_last_lr()[0]:.4f} | Scaler: {self.scaler.get_scale()} | Norm: {norm:.4f}\n")

        return loss.detach().item()

    @torch.no_grad
    def val_step(self, batch, debug: bool = False):
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
                ctrl_input_ids = encoded_batch["ctrl_input_ids"],
                ctrl_attention_mask = encoded_batch["ctrl_attention_mask"],
                edited_input_ids = encoded_batch["edited_input_ids"],
                edited_attention_mask = encoded_batch["edited_attention_mask"],
            )
            loss = self.loss_func(outputs, encoded_batch["targets"])

        if debug:
            print(f"\nValidation | Loss: {loss.item():.4f}")

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
                ctrl_input_ids = encoded_batch["ctrl_input_ids"],
                ctrl_attention_mask = encoded_batch["ctrl_attention_mask"],
                edited_input_ids = encoded_batch["edited_input_ids"],
                edited_attention_mask = encoded_batch["edited_attention_mask"],
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
            for batch in tqdm(self.train_loader):
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
        preds = [] #[n_cells, n_genes]
        gts = []
        perturbations = []
        covariates = []
        eval_losses = []
        
        bidx = 0
        for batch in tqdm(loader):
            loss, outputs, targets = self.test_step(batch, debug=debug)
            eval_losses.append(loss)

            preds.append(outputs.detach().float().cpu().numpy())
            gts.append(targets.detach().float().cpu().numpy())
            perturbations.extend(batch["perturbations"])
            covariates.extend(batch["covariates"])

            # bidx += 1

            # if bidx >= 1:
            #     break

        preds = np.concatenate(preds, axis=0)
        gts = np.concatenate(gts, axis=0)

        assert preds.shape == gts.shape, f"Predictions shape {preds.shape} does not match ground truth shape {gts.shape}"
        eval_loss = np.mean(eval_losses)
        print(f"Evaluation Loss: {eval_loss}")

        print("Evaluating perturbation response prediction metrics...")
        print(f"preds shape: {preds.shape}, gts shape: {gts.shape}")
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
    
        