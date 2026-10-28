import torch
from nanomirrors.data import load_dataset
from nanomirrors.auto import AutoTrainerForPerturbationResponsePrediction

data = load_dataset("adamson2016") # load preprocessed perturbation dataset from HF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")

# Automatically intialize model, configs, and corresponding trainer 
trainer = AutoTrainerForPerturbationResponsePrediction.from_model(
    model_name="State",
    n_genes=data.shape[1], # number of genes in data
    device=device
)

history = trainer.train(train_data=data, val_data=None, checkpoint_path="./run")
