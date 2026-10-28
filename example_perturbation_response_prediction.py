import torch
from nanomirrors.data import load_dataset
from nanomirrors.auto import AutoTrainerForPerturbationResponsePrediction

data = load_dataset("adamson2016") # load preprocessed perturbation dataset from HF

# Automatically intialize model, configs, and corresponding trainer 
trainer = AutoTrainerForPerturbationResponsePrediction.from_model(
    model_name="State",
    n_genes=data.shape[1], # number of genes in data
    device=torch.device("cuda")
)

history = trainer.train(train_data=data, val_data=None, checkpoint_path="./run")
