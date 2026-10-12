from dataclasses import asdict, dataclass
from typing import Optional, Tuple
from tqdm import tqdm
from pathlib import Path
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import anndata as ad

from nanomirrors.data.perturbation_dataset import PairedPerturbationDataset
from nanomirrors.models.scgpt.input_encoding_scgpt import scGPTInputEncoderForPerturbation

@dataclass
class scGPTTrainerConfig:
    lr: float = 1e-4
    max_steps: int = 10000
    batch_size: int = 64
    amp: bool = True # Use automatic mixed precision during training
    scheduler_gamma: float = 0.9

    early_stopping_patience: int = 2000
    gradient_clip_norm: float = 1.0
    val_every: int = 2000

    log_every: int = 100

class scGPTTrainerForPerturbationResponsePrediction:

    def __init__(self, 
                 model: nn.Module,
                 train_data: ad.AnnData,
                 val_data: ad.AnnData | None,
                 config: scGPTTrainerConfig,
                 device: torch.device,
                 pair_covariates: list[str] | None = None):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self.device = device
        self.pair_covariates = pair_covariates
        
        # Initialize the model-task specific dataset, dataloaders.
        self.train_dataset = PairedPerturbationDataset(train_data, pair_covariates=pair_covariates)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)

        batch = next(iter(self.train_loader))
        print(batch)
        
        if val_data is not None:
            self.val_dataset = PairedPerturbationDataset(val_data, pair_covariates=pair_covariates)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Initialize the input encoder.
        self.input_encoder = scGPTInputEncoderForPerturbation(
            gene_names = train_data.var['gene_symbol'].tolist(),
            model_name = model.config.model_name,
            truncate_to_max_length=True,
        )
        
        self.loss_fn = nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                                  lr=self.config.lr)
        scheduler_step_size = len(self.train_loader) # scGPT step decay LR per one epoch.
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
        
    def loss_func(self, outputs, targets):
        loss = self.loss_fn(outputs, targets) # [B, T] -> [B] -> scalar

        return loss
    
    def train_step(self, batch, debug=False) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        encoded_batch = self.input_encoder.encode(batch)
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
            print(
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

        encoded_batch = self.input_encoder.encode(batch)
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
    def test_step(self, batch, input_encoder, debug: bool = False) -> Tuple[float, torch.Tensor]:
        self.model.eval()

        encoded_batch = input_encoder.encode(batch)
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

    def train(self, checkpoint_path: Optional[str] = None, debug: bool = False):
        history = []
        while self.global_step < self.config.max_steps:   
            for batch in tqdm(self.train_loader, desc=f"Training Step {self.global_step}", leave=False):
                loss = self.train_step(batch, debug=debug)
                
                metrics = {"step": self.global_step, "train_loss": loss}
    
                report = self.global_step == 1 or self.global_step % self.config.log_every == 0
                
                if self.global_step % self.config.val_every == 0:
                    if self.val_loader is not None:
                        metrics["val_loss"] = self.evaluate(self.val_loader, debug=debug)
                        report = True
                    if checkpoint_path is not None:
                        self.save_checkpoint(checkpoint_path)

                    # !TODO: Implement early stopping based on pearson correlation across genes.
                    # !NOTE: can also use pearson delta instead.
                    # metric_name = "pearson"
                    # if metrics[metric_name] > self.best_val_metric:
                    #     self.best_val_metric = metrics[metric_name]
                    #     torch.save(self.model.state_dict(), f"{self.save_dir}/best_model.pt")
                    #     print(f"New best model saved with {metric_name}: {self.best_val_metric:.4f}")
                    #     self.patience = 0
                    # else:
                    #     self.patience += 1
                    #     if self.patience >= self.early_stopping_patience:
                    #         print(f"Early stopping triggered after {self.global_step} steps.")
                    #         break    
                    
                if report or self.global_step == self.config.max_steps:
                    history.append(metrics)
                    print(f"{'| '.join(f'{k}: {v}' for k, v in metrics.items())}")
                
                if self.global_step >= self.config.max_steps:
                    break
        
        if checkpoint_path is not None:
            self.save_checkpoint(checkpoint_path)
        
        return history

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
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint["model_config"] != asdict(self.model.config):
            raise ValueError("Checkpoint model configuration does not match this model")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint["global_step"]
    