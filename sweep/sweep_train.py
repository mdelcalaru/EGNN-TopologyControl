import os
import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import EarlyStopping

from graph_models.graph_model import EGNN_network
from graph_models.callbacks import SmoothedLossMonitor
from graph_models.graph_training_utils import load_datasets_flexible

def train_sweep():
    run = wandb.init()
    config = run.config

    output_dims = config.dim_pairs[0]
    m_dims = config.dim_pairs[1]

    # Data loading (igual que antes)
    train_data, val_data, _, _ = load_datasets_flexible(
        directory="./data_sync/datasets/fix_15_agents/",
        min_nodes=1 if not config.use_all_data else None,
        max_nodes=20 if not config.use_all_data else None,
        use_all=config.use_all_data,
        filter_threshold=config.filter_threshold
    )
    train_loader = DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True, num_workers=int(2*os.cpu_count()/3)
    )
    val_loader = DataLoader(
        val_data, batch_size=config.batch_size, shuffle=False, num_workers=int(2*os.cpu_count()/3)
    )

    # Prepara dict limpio para logger y modelo
    config_dict = {k: v for k, v in config.items()}
    config_dict['output_dims'] = output_dims
    config_dict['m_dims'] = m_dims

    model = EGNN_network(**config_dict)

    smoothed_cb = SmoothedLossMonitor(window_size=4)
    early_stopping = EarlyStopping(monitor="smoothed_val_loss", patience=10, mode="min", verbose=False)
# Esto solo se puede hacer si wandb ya se ha inicializado antes

    project_name = wandb.run.project

    # Logger con dict limpio en config
    logger = WandbLogger(project=project_name, log_model=True, config=config_dict)

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=logger,
        callbacks=[smoothed_cb, early_stopping],
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
    )

    trainer.fit(model, train_loader, val_loader)

    run.finish()



if __name__ == "__main__":
    train_sweep()
