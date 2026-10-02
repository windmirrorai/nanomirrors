from dataclasses import asdict, dataclass
from typing import Optional
from dataclasses import field
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import anndata as ad
from geomloss import SamplesLoss

from nanomirrors.data.perturbation_dataset import PairedSetPerturbationDataset
from .input_encoding_state import StateInputEncoderForPerturbationResponsePrediction

"""
trainer.train(train_data, val_data)
    - dataset, loader
    - input encoder
    - train_step
predictions = trainer.predict(test_data)
"""

@dataclass
class StateTrainerConfig:
    use_multi_hot: bool = True
    use_pert_features: bool = False
    perturbation_features_path: str | None = None

    lr: float = 1e-3
    max_steps: int = 80000
    batch_size: int = 32
    amp: bool = False # Use automatic mixed precision during training

    gradient_clip_norm: float = 10.0
    val_every: int = 2000

    log_every: int = 100
    device: str = "auto"

    set_covariates: list[str] = field(default_factory=lambda: ["perturbation", "cell_line"])

class StateTrainerForPerturbationResponsePrediction:

    def __init__(self, 
                 model: nn.Module,
                 train_data: ad.AnnData,
                 val_data: ad.AnnData | None,
                 config: StateTrainerConfig,
                 device: torch.device):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self.device = device

        self.cell_set_size = model.config.cell_set_size

        # Initialize the model-task specific dataset, dataloaders.
        for cov in self.config.set_covariates:
            if cov not in train_data.obs.columns:
                raise ValueError(f"Covariate '{cov}' not found in training data. \
                                 Please ensure it is included in the AnnData object's obs.")
        self.train_dataset = PairedSetPerturbationDataset(train_data, 
                                                          set_size=self.cell_set_size,
                                                          set_covariates=self.config.set_covariates)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)

        if val_data is not None:
            for cov in self.config.set_covariates:
                if cov not in val_data.obs.columns:
                    raise ValueError(f"Covariate '{cov}' not found in validation data. \
                                    Please ensure it is included in the AnnData object's obs.")
            self.val_dataset = PairedSetPerturbationDataset(val_data, 
                                                            set_size=self.cell_set_size,
                                                            set_covariates=self.config.set_covariates)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False)


        # Initialize the input encoder.
        perturbation_vocab = self.train_dataset.perturbations
        self.input_encoder = StateInputEncoderForPerturbationResponsePrediction(
            perturbation_vocab=perturbation_vocab,
            perturbation_features=None,
            use_multi_hot=self.config.use_multi_hot,
            use_pert_features=self.config.use_pert_features,
        )

        self.loss_fn = SamplesLoss(loss="energy")
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.config.lr)
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
        loss = self.loss_fn(outputs, targets).mean(axis=0) # [B, S, D]

        return loss

    def train_step(self, batch):
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
            enabled=self.amp):
            outputs = self.model(
                basal_exprs=encoded_batch["basal_exprs"],
                perts=encoded_batch["perts"],
                batch=encoded_batch.get("batch", None)
            )
            loss = self.loss_func(outputs, encoded_batch["targets"])

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        self.global_step += 1
        return loss.detach().item()

    @torch.no_grad
    def val_step(self, batch):
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
                basal_exprs=encoded_batch["basal_exprs"],
                perts=encoded_batch["perts"],
                batch=encoded_batch.get("batch", None)
            )
            loss = self.loss_func(outputs, encoded_batch["targets"])

        return loss.detach().item()

    def train(self, checkpoint_path: Optional[str] = None):
        history = []
        while self.global_step < self.config.max_steps:   
            for batch in tqdm(self.train_loader, desc=f"Training Step {self.global_step}", leave=False):
                loss = self.train_step(batch)
                
                metrics = {"step": self.global_step, "train_loss": loss}
    
                report = self.global_step == 1 or self.global_step % self.config.log_every == 0
                
                if self.global_step % self.config.val_every == 0:
                    if self.val_loader is not None:
                        metrics["val_loss"] = self.evaluate(self.val_loader)
                        report = True
                    if checkpoint_path is not None:
                        self.save_checkpoint(checkpoint_path)
                
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

    