
from graph_model import LightningEGNN_net
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import os
from graph_dataset_utils import build_dataset
import torch
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_directory = "data_nuevo_canal/sinNorm/"#/data/4N/"
cpus = os.cpu_count()

config_dict = { 
    'optimizer': 'adam',
    'output_dims': [4 ,8],
    'm_dims': [16,32],
    'update_feats': True,
    'update_coors': False,
    'learning_rate': 0.005, 
    'dropout': 0.0,
    'batch_size': 20, 
    'epochs': 200,  
    'NormCoors':False, 
    'aggr':'add',  
    }

'''Load Model'''

#artifact_file="./model/EGNN_best_model.ckpt"
#model_file = Path(artifact_file)
artifact_name="model-ctt982k5:v99"
#  artifact_name="model-nmj8b857:v19" 
artifact_dir="./artifacts/" + artifact_name +"/model.ckpt"
model_file = Path(artifact_dir)

if not model_file.exists():
    import wandb
    run = wandb.init()
    artifact = run.use_artifact("iie-sc/new_EGNN/"+artifact_name , type='model')
    artifact_dir = artifact.download()    


model =LightningEGNN_net.load_from_checkpoint(model_file)                           
                                                 


#model = LightningEGNN_net(**config_dict)
#print(model)

logger = pl_loggers.WandbLogger(log_model= 'all' ,project="new_EGNN", config=config_dict)
train_loader, test_loader = build_dataset(batch_size=config_dict['batch_size'], dir=data_directory, cpus=cpus)
trainer = pl.Trainer(logger=logger, max_epochs=config_dict['epochs']  )
#trainer.fit(model, train_loader, test_loader)
trainer.fit(model, train_loader, test_loader,ckpt_path=artifact_dir)
#model-p5qvg23t:v9
'''
#trainer.fit(model, train_loader, test_loader,ckpt_path=model_file)
out_train, out_test = build_dataset(batch_size=config_dict['batch_size'], dir="./data/Ind_graph_2/", cpus=cpus)

print(trainer.test(model, dataloaders=out_test))
print(trainer.test(model, dataloaders=out_train))'''