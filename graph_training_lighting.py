
from graph_model import LightningEGNN_net
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import os
from graph_dataset_utils import build_dataset
import torch
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_directory = "./data/"
cpus = os.cpu_count()

config_dict = {
    'optimizer': 'adam',
    'output_dims': [16,32,21],
    'm_dims': [16,32,64],
    'update_coors': False,
    'learning_rate': 0.001,
    'dropout': 0.0,
    'batch_size': 20, 
    'epochs': 10,  
    'NormCoors':False, 
    'aggr':'mean',  
    }

model = LightningEGNN_net(**config_dict)
#print(model)

logger = pl_loggers.WandbLogger(log_model= 'all' ,project="EGNN", config=config_dict)
train_loader, test_loader = build_dataset(batch_size=config_dict['batch_size'], dir=data_directory, cpus=cpus)
trainer = pl.Trainer(logger=logger, max_epochs=config_dict['epochs']  )
trainer.fit(model, train_loader, test_loader)


