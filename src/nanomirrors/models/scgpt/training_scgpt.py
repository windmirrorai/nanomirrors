from dataclasses import asdict, dataclass
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm
from pathlib import Path
import warnings
import logging

import os

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import anndata as ad

from nanomirrors.data.perturbation_dataset import PairedPerturbationDataset
from nanomirrors.models.scgpt.input_encoding_scgpt import scGPTInputEncoderForPerturbation
from nanomirrors.evaluation import compute_perturbation_metrics

@dataclass
class scGPTTrainerConfig:
    lr: float = 1e-4
    max_steps: int = 10000
    batch_size: int = 64
    amp: bool = True # Use automatic mixed precision during training
    scheduler_gamma: float = 0.9

    early_stopping_patience: int = 20 # Number of validation steps to wait before early stopping
    gradient_clip_norm: float = 1.0

    val_every: int = 50
    log_every: int = 20

class scGPTTrainerForPerturbationResponsePrediction:

    def __init__(self, 
                 model: nn.Module,
                 config: scGPTTrainerConfig,
                 device: torch.device,
                 logger: Optional[logging.Logger] = None,
                 save_dir: str = "./"):
        self.model = model
        self.config = config
        self.device = device
        self.train_input_encoder = None
        self.eval_input_encoder = None
        self.logger = logger
        self.save_dir = save_dir
        
        self.loss_fn = nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                                  lr=self.config.lr)
        self.scheduler = None

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
    
    def train_step(self, batch, debug=False) -> float:
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
            enabled=self.amp
        ):
            outputs = self.model(
               gene_tokens = encoded_batch["gene_tokens"],
               exprs = encoded_batch["exprs"],
               padding_mask = encoded_batch["padding_mask"],
               pert_labels = encoded_batch["pert_labels"],
            ) # [B, T]

            valid = ~encoded_batch["padding_mask"]
            loss = self.loss_func(outputs[valid], encoded_batch["targets"][valid])

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
            print(f"Step {self.global_step} | Loss: {loss.item():.4f} | LR: {self.scheduler.get_last_lr()[0]:.4f} | Scaler: {self.scaler.get_scale()} | Norm: {norm:.4f}")
        
        return loss.detach().item()

    @torch.no_grad
    def val_step(self, batch, debug: bool = False) -> float:
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
                gene_tokens = encoded_batch["gene_tokens"],
                exprs = encoded_batch["exprs"],
                padding_mask = encoded_batch["padding_mask"],
                pert_labels = encoded_batch["pert_labels"],
            )
            valid = ~encoded_batch["padding_mask"]
            loss = self.loss_func(outputs[valid], encoded_batch["targets"][valid])

        if debug:
            print(f"Validation | Loss: {loss.item():.4f}")

        return loss.detach().item()

    @torch.inference_mode()
    def test_step(self, batch, debug: bool = False) -> Tuple[float, torch.Tensor]:
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
                gene_tokens = encoded_batch["gene_tokens"],
                exprs = encoded_batch["exprs"],
                padding_mask = encoded_batch["padding_mask"],
                pert_labels = encoded_batch["pert_labels"],
            )
            valid = ~encoded_batch["padding_mask"]
            loss = self.loss_func(outputs[valid], encoded_batch["targets"][valid])

        if debug:
            print(f"Test Loss: {loss.item():.4f}")

        return loss.detach().item(), outputs, encoded_batch["targets"]

    def train(self,
              train_data: ad.AnnData,
              val_data: Optional[ad.AnnData] = None, 
              checkpoint_path: Optional[str] = None, 
              debug: bool = False) -> Dict[str, List[Dict[str, float]]]:

        if "covariates" not in train_data.uns:
            raise ValueError("Covariates not found in  `train_data.uns['covariates']`. Please ensure that the training data contains covariates.")
        if val_data is not None and "covariates" not in val_data.uns:
            raise ValueError("Covariates not found in  `val_data.uns['covariates']`. Please ensure that the validation data contains covariates.")      


        train_dataset = PairedPerturbationDataset(train_data, 
                                                       pair_covariates=train_data.uns["covariates"], 
                                                       include_ctrl_ctrl_pairs=True)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        if val_data is not None:
            val_dataset = PairedPerturbationDataset(val_data, 
                                                         pair_covariates=val_data.uns["covariates"], 
                                                         include_ctrl_ctrl_pairs=False)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

        # Initialize the input encoder(s).
        self.train_input_encoder = scGPTInputEncoderForPerturbation(
            gene_names = train_data.var['gene_symbol'].tolist(),
            model_name = self.model.config.model_name,
            truncate_to_max_length=True,
        )
        self.eval_input_encoder = scGPTInputEncoderForPerturbation(
            gene_names = train_data.var['gene_symbol'].tolist(),
            model_name = self.model.config.model_name,
            truncate_to_max_length=False,
        )

        scheduler_step_size = len(train_loader) # scGPT step decay LR per one epoch.
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                         step_size=scheduler_step_size, 
                                                         gamma=self.config.scheduler_gamma)
        
        train_losses = []
        val_losses = []
        metrics_history = []

        self.best_val_metric = float("-inf")
        metric_category= "expression"
        metric_name = "pearson_delta_all_genes"
        self.patience = 0

        stop_training = False
        pbar = tqdm(
            total=self.config.max_steps,
            initial=self.global_step,
            desc="Training",
            unit="step",
        )
        while not stop_training:
            for batch in train_loader:
                prev_global_step = self.global_step
                loss = self.train_step(batch, debug=debug)
                train_losses.append({"step": self.global_step, "train_loss": loss})

                pbar.update(self.global_step - prev_global_step)
                pbar.set_postfix(loss=f"{loss:.4f}")

                if self.global_step % self.config.log_every == 0 or self.global_step == 1:
                    self._log(f"Step {self.global_step} | Train Loss: {loss:.4f}")
    
                '''
                Perform validation every self.config.val_every steps.
                '''
                if self.global_step % self.config.val_every == 0:
                    if val_loader is None:
                        self._log("[Warning] Validation loader is not provided. No checkpoint will be saved.")
                    else:
                        val_loss, val_results, _, _, _, _ = self.evaluate(loader=val_loader, debug=debug)
                        val_losses.append({"step": self.global_step, "val_loss": val_loss})
                        metrics_history.append({"step": self.global_step, **val_results}) 

                        # Log the current step and validation results.
                        self._log(f"Step {self.global_step}")
                        for k, v in val_results.items():
                            self._log(f"{k}: {v}")

                        # Save best model based on the specified validation metric.
                        if val_results[metric_category][metric_name] > self.best_val_metric:
                            self.best_val_metric = val_results[metric_category][metric_name]
                            self.save_checkpoint(checkpoint_path)
                            self._log(f"New best model saved with {metric_name}: {self.best_val_metric:.4f}")
                            self.patience = 0
                        else:
                            self.patience += 1
                            if self.patience >= self.config.early_stopping_patience:
                                self._log(f"Stopping training at step {self.global_step} due to early stopping.")
                                stop_training = True
                                break

                if self.global_step >= self.config.max_steps:
                    self._log(f"Maximum training steps reached: {self.global_step}, stopping training.")
                    stop_training = True
                    break
        
        pbar.close()

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
        for batch in tqdm(loader, desc="Evaluating"):
            loss, outputs, targets = self.test_step(batch, debug=debug)
            eval_losses.append(loss)

            preds.append(outputs.detach().float().cpu().numpy())
            gts.append(targets.detach().float().cpu().numpy())
            perturbations.extend(batch["perturbations"])
            covariates.extend(batch["covariates"])

            # if bidx % 10 == 0:
            #     print(
            #         f"Batch {bidx} | "
            #         f"RSS: {process.memory_info().rss / 1e9:.2f} GB | "
            #         f"Available RAM: {psutil.virtual_memory().available / 1e9:.2f} GB",
            #         flush=True,
            #     )

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
    
    def test(self, data: ad.AnnData, debug=False):
        if 'covariates' not in data.uns:
            raise ValueError("The AnnData object must contain 'covariates' in its .uns attribute")
            
        dataset = PairedPerturbationDataset(data, pair_covariates=data.uns['covariates'], include_ctrl_ctrl_pairs=False)
        test_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        if not hasattr(self, "eval_input_encoder") or self.eval_input_encoder is None:
            # Initialize the test input encoder, in case it has not been initialized during training.
            self.eval_input_encoder = scGPTInputEncoderForPerturbation(
                gene_names = data.var['gene_symbol'].tolist(),
                model_name = self.model.config.model_name,
                truncate_to_max_length = False,
            )
        
        return self.evaluate(loader=test_loader, debug=debug)

    def save_checkpoint(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "model_config": asdict(self.model.config),
            "trainer_config": asdict(self.config),
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint["model_config"] != asdict(self.model.config):
            print(
                "[Warning] Checkpoint model configuration does not match this model. "
                "This may lead to unexpected behavior if the model configuration differs from the checkpoint."
            )
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint["global_step"]
        self.config = scGPTTrainerConfig(**checkpoint["trainer_config"])
