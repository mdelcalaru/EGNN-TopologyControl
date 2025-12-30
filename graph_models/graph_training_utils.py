import os
import re
import json
import pickle as pk
import torch
from pathlib import Path
from typing import List, Tuple
from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import wandb
from graph_models.graph_model import EGNN_network
from graph_models.callbacks import SmoothedLossMonitor, SaveWandbInfoCallback, LogGradNormCallback


def get_node_count_from_filename(filename: str) -> int:
    """Extracts the number of agents from a filename based on the 'agentsXXX_' pattern."""
    match = re.search(r'agents(\d+)_', filename)
    return int(match.group(1)) if match else None


def load_files_by_type_and_range(
    files: List[Path], min_nodes: int, max_nodes: int
) -> Tuple[List[torch.utils.data.Dataset], List[str]]:
    """
    Loads datasets from files where the number of agents (nodes) is within a given range.

    Returns:
        - List of loaded data objects.
        - List of filenames used.
    """
    dataset = []
    used_files = []
    for file in sorted(files):
        node_count = get_node_count_from_filename(file.name)
    
        if node_count is not None and min_nodes <= node_count <= max_nodes:
            with open(file, 'rb') as f:
                data_loaded = pk.load(f)
                dataset.extend(data_loaded)
                used_files.append(file.name)
        elif node_count is None and (min_nodes == float('-inf') or max_nodes == float('inf')):
            with open(file, 'rb') as f:
                data_loaded = pk.load(f)
                dataset.extend(data_loaded)
                used_files.append(file.name)
                
    return dataset, used_files


def load_datasets_flexible(
    directory: str,
    min_nodes: int = None,
    max_nodes: int = None,
    use_all: bool = False,
    filter_threshold: float = None
) -> Tuple[List[torch.utils.data.Dataset], List[torch.utils.data.Dataset], List[str], List[str]]:
    """
    Loads training and validation datasets from a directory, with options to:
    - Select only samples within a given node range.
    - Load all available data regardless of node count.
    - Apply an optional filter on the mean absolute `dif_y` value.

    Args:
        directory (str): Path to the dataset directory.
        min_nodes (int, optional): Minimum number of nodes. Ignored if use_all is True.
        max_nodes (int, optional): Maximum number of nodes. Ignored if use_all is True.
        use_all (bool): If True, loads all data regardless of node count.
        filter_threshold (float, optional): If set, filters samples with mean(abs(dif_y)) < threshold.

    Returns:
        Tuple containing:
            - train_dataset: List of training samples.
            - val_dataset: List of validation samples.
            - train_files_used: List of training file names used.
            - val_files_used: List of validation file names used.
    """
    directory = Path(directory)
    train_files = list(directory.glob("*_train.pt"))
    val_files = list(directory.glob("*_val.pt"))
    if use_all:
        min_nodes = float('-inf')
        max_nodes = float('inf')

    train_dataset, train_files_used = load_files_by_type_and_range(train_files, min_nodes, max_nodes)
    val_dataset, val_files_used = load_files_by_type_and_range(val_files, min_nodes, max_nodes)

    if filter_threshold is not None:
        train_dataset = [
            data for data in train_dataset if torch.mean(torch.abs(data.dif_y)) >= filter_threshold
        ]
        val_dataset = [
            data for data in val_dataset if torch.mean(torch.abs(data.dif_y)) >= filter_threshold
        ]

    return train_dataset, val_dataset, train_files_used, val_files_used


def train_model_with_configurable_data(
    model, config, logger, smoothed_cb,
    node_range, suffix, data_directory, project_name, CPUS,
    use_all_data=False, filter_threshold=None
):
    """
    Trains a model using a configurable dataset loader, allowing selection by node count
    and filtering by target difference (`dif_y`).

    Args:
        model: The PyTorch Lightning model to train.
        config (dict): Configuration dictionary for the model and training.
        logger: WandbLogger instance.
        smoothed_cb: Smoothed loss callback to monitor validation loss.
        node_range (tuple): Tuple (min_nodes, max_nodes). Ignored if use_all_data is True.
        suffix (str): Suffix to identify the saved checkpoint.
        data_directory (str): Path to the dataset folder.
        project_name (str): Name of the W&B project.
        CPUS (int): Number of CPU workers for data loading.
        use_all_data (bool): If True, ignores node_range and loads all nodes available data.
        filter_threshold (float, optional): If set, filters data with low dif_y signal.

    Saves:
        - A model checkpoint in the "checkpoints/" directory.
    """
    if use_all_data:
        print(f"\nTraining with ALL node data.")
    else:
        print(f"\nTraining with nodes in range: {node_range}")

    train_data, val_data, train_files_used, val_files_used = load_datasets_flexible(
        directory=data_directory,
        min_nodes=node_range[0] if not use_all_data else None,
        max_nodes=node_range[1] if not use_all_data else None,
        use_all=use_all_data,
        filter_threshold=filter_threshold
    )

    train_size, val_size = len(train_data), len(val_data)
    print(f"Train size: {train_size} | Val size: {val_size}")

    new_config = {
        "dataset/directory": str(data_directory),
        "dataset/used_all_data": use_all_data,
        "dataset/node_range": f"{node_range[0]} to {node_range[1]}" if not use_all_data else "ALL",
        "dataset/filter_threshold": filter_threshold if filter_threshold is not None else "None",
        "dataset/train_samples": train_size,
        "dataset/val_samples": val_size,
        "dataset/train_files_used": ", ".join(train_files_used),
        "dataset/val_files_used": ", ".join(val_files_used)        
    }
    new_config.update({
    "training/epochs": config['epochs'],
    "training/batch_size": config['batch_size'],
    "training/gradient_clip_val": config.get("gradient_clip_val", None),
    "training/gradient_clip_algorithm": config.get("gradient_clip_algorithm", "norm"),
    })

    logger.experiment.config.update(new_config, allow_val_change=True)

    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=CPUS)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False, num_workers=CPUS)

    early_stopping = EarlyStopping(monitor='smoothed_val_loss', patience=5, min_delta=0.001, verbose=True, mode='min')

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config['epochs'],
        callbacks=[smoothed_cb, early_stopping,LogGradNormCallback(log_every_n_steps=20)],
        gradient_clip_val=config.get("gradient_clip_val", None),  
        gradient_clip_algorithm=config.get("gradient_clip_algorithm", "norm")
    )

    trainer.fit(model, train_loader, val_loader)

    Path("checkpoints").mkdir(exist_ok=True)
    ckpt_name = f"checkpoints/{project_name}_{logger.experiment.name}_model_{suffix}.ckpt"
    trainer.save_checkpoint(ckpt_name)
    print(f"Saved checkpoint: {ckpt_name}")



def check_if_weights_changed(model, initial_weights):
    """
    Compares model parameters before and after training.

    Args:
        model: The trained model.
        initial_weights (dict): Dictionary of parameter values before training.

    Returns:
        bool: True if any parameter changed, False otherwise.
    """
    for name, param in model.named_parameters():
        if not torch.equal(param, initial_weights[name]):
            return True
    return False

def train_with_curriculum(model, 
                          config, 
                          node_curriculum, 
                          logger, 
                          smoothed_cb, 
                          data_directory,
                          project_name,
                          CPUS,
                          use_all_data=False,
                          filter_threshold=None):
    
    for stage_idx, node_range in enumerate(node_curriculum, start=1):
        print(f"\n=== Etapa {stage_idx}/{len(node_curriculum)}: Entrenando con nodos {node_range} ===")
        suffix = f"stage{stage_idx}_nodes{node_range[0]}-{node_range[1]}"
        # Entrenamiento parcial
        train_model_with_configurable_data(
            model=model,
            config=config,
            logger=logger,
            smoothed_cb=smoothed_cb,
            node_range=node_range,
            suffix=suffix,
            data_directory=data_directory,
            project_name=project_name,
            CPUS=CPUS,
            use_all_data=use_all_data,
            filter_threshold=filter_threshold
        )

    print("\n🎯 Entrenamiento con curriculum finalizado.")
