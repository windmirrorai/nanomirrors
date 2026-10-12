from dataclasses import asdict, dataclass
from typing import Optional, Tuple
from dataclasses import field
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import anndata as ad

from nanomirrors.data.perturbation_dataset import PairedPerturbationDataset
from nanomirrors.models.geneformer.input_encoding_geneformer import GeneformerInputEncoderForPerturbation

@dataclass
class GeneformerTrainerConfig:
    lr: float = 1e-3
    max_steps: int = 80000
    batch_size: int = 32
    amp: bool = True # Use automatic mixed precision during training

    gradient_clip_norm: float = 10.0
    val_every: int = 2000

    log_every: int = 100

class GeneformerTrainerForPerturbationResponsePrediction:

    def __init__(self, 
                 model: nn.Module,
                 train_data: ad.AnnData,
                 val_data: ad.AnnData | None,
                 config: GeneformerTrainerConfig,
                 device: torch.device):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self.device = device
        
        # Initialize the model-task specific dataset, dataloaders.
        self.train_dataset = PairedPerturbationDataset(train_data)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)

        batch = next(iter(self.train_loader))
        print(batch)
        
        if val_data is not None:
            self.val_dataset = PairedPerturbationDataset(val_data)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        
        # Initialize the input encoder.
        perturbation_vocab = self.train_dataset.perturbations
        if "v2" in model.config.model_name:
             tokenizer_name = "geneformer_V2"
        elif "v1" in model.config.model_name:
             tokenizer_name = "geneformer_V1"
        else:
             raise ValueError(f"Unknown model name: {model.config.model_name}")
        self.input_encoder = GeneformerInputEncoderForPerturbation(
            gene_names = train_data.var['gene_ensembl'].dropna().tolist(),
            perturbation_method = "CRISPRi",
            model_name = tokenizer_name
        )
        
        self.loss_fn = nn.MSELoss(reduction="mean")
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
        loss = self.loss_fn(outputs, targets) # [B, T] -> [B] -> scalar

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
                ctrl_input_ids = encoded_batch["ctrl_input_ids"],
                ctrl_attention_mask = encoded_batch["ctrl_attention_mask"],
                edited_input_ids = encoded_batch["edited_input_ids"],
                edited_attention_mask = encoded_batch["edited_attention_mask"]
            ) # [B, T]

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
                ctrl_input_ids = encoded_batch["ctrl_input_ids"],
                ctrl_attention_mask = encoded_batch["ctrl_attention_mask"],
                edited_input_ids = encoded_batch["edited_input_ids"],
                edited_attention_mask = encoded_batch["edited_attention_mask"],
            )
            loss = self.loss_func(outputs, encoded_batch["targets"])

        return loss.detach().item()

    @torch.inference_mode()
    def test_step(self, batch, input_encoder, debug: bool = False) -> Tuple[float, torch.Tensor, torch.Tensor]:
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
                ctrl_input_ids = encoded_batch["ctrl_input_ids"],
                ctrl_attention_mask = encoded_batch["ctrl_attention_mask"],
                edited_input_ids = encoded_batch["edited_input_ids"],
                edited_attention_mask = encoded_batch["edited_attention_mask"],
            )
            loss = self.loss_func(outputs, encoded_batch["targets"])
        
        return loss.detach().item(), outputs, encoded_batch["targets"]

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
    
        