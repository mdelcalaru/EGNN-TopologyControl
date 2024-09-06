import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms
from graph_dataset_utils import build_dataset
from graph_model import LightningEGNN_net
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
import wandb
import os
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device="cpu"
with open('config_ggn_sweeps.yaml', 'r') as file:
    sweep_config = yaml.safe_load(file)

sweep_config['parameters'].update({'cpus':{'value': os.cpu_count()}})




def train(config=None):
    # Initialize a new wandb run
    wandb.init()
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller 
    config = wandb.config
    train_loader, test_loader = build_dataset(batch_size=config.batch_size, dir=config.data_directory, cpus=config.cpus)
    network = LightningEGNN_net(**config) 
    logger = pl_loggers.WandbLogger(log_model=False,project="new_EGNN")
    trainer = pl.Trainer(logger=logger, max_epochs=wandb.config.epochs)
    trainer.fit(network, train_loader, test_loader)

sweep_id = "5tdkfszp"#wandb.sweep(sweep_config, project="new_EGNN")
wandb.agent(sweep_id, train, count=4,project="new_EGNN")

