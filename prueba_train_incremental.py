
from tabnanny import check
from graph_model import LightningEGNN_net
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import os
from graph_dataset_utils import build_dataset
import torch
from pathlib import Path
from data_loading import customDataModule

config_dict = {
    'optimizer': 'adam',
    'output_dims': [32,32],
    'm_dims': [32,32],
    'update_feats': True,
    'update_coors': False,
    'learning_rate': 0.001,
    'dropout': 0.0,
    'batch_size': 10, 
    'epochs': 10,  
    'NormCoors':False, 
    'aggr':'mean',  
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sub_dir=['10N/','20N/','30N/','40N/','50N/']
data_directory = "./data/"
cpus = os.cpu_count()
data_loader=customDataModule(batch_size=config_dict['batch_size'], cpus=cpus, dir=data_directory, subdirs=sub_dir, change_interval=10)
model = LightningEGNN_net(**config_dict)
logger = pl_loggers.WandbLogger(log_model= 'all' ,project="new_EGNN", config=config_dict)
#train_loader, test_loader = build_dataset(batch_size=config_dict['batch_size'], dir=data_directory+sub_dir[0], cpus=cpus)
trainer = pl.Trainer(logger=logger, max_epochs=config_dict['epochs']*len(sub_dir))
trainer.fit(model, data_loader.train_loader, data_loader.val_loader)

