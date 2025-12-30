import glob
import pickle as pk
import re
import torch
from torch_geometric.loader import DataLoader
from pathlib import Path
from typing import List, Tuple

def get_node_count_from_filename(filename: str) -> int:
    """Extrae la cantidad de agentes del nombre del archivo."""
    match = re.search(r'agents(\d+)_', filename)
    return int(match.group(1)) if match else None

def load_files_by_type_and_range(
    files: List[Path], min_nodes: int, max_nodes: int
) -> Tuple[List[torch.utils.data.Dataset], List[str]]:
    """
    Carga datasets desde archivos cuya cantidad de nodos esté dentro del rango.
    Devuelve la lista de datos y la lista de nombres de archivos usados.
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
    return dataset, used_files

def load_datasets_by_node_range(
    directory: str,
    min_nodes: int,
    max_nodes: int,
    filtered: bool = False
) -> Tuple[List[torch.utils.data.Dataset], List[torch.utils.data.Dataset], List[str], List[str]]:
    """
    Carga datasets de entrenamiento y validación según rango de nodos.
    También devuelve el listado de archivos usados en cada conjunto.
    """
    directory = Path(directory)
    train_files = list(directory.glob("*_train.pt"))
    val_files = list(directory.glob("*_val.pt"))

    train_dataset, train_files_used = load_files_by_type_and_range(train_files, min_nodes, max_nodes)
    val_dataset, val_files_used = load_files_by_type_and_range(val_files, min_nodes, max_nodes)

    if filtered:
        train_dataset = [
            data for data in train_dataset if torch.mean(torch.abs(data.dif_y)) >= 0.02
        ]
        val_dataset = [
            data for data in val_dataset if torch.mean(torch.abs(data.dif_y)) >= 0.02
        ]

    return train_dataset, val_dataset, train_files_used, val_files_used



### dataset builders for training with pytorch ###
def build_dataset(batch_size, cpus, dir):
    #this function asumes that the data is stored in a directory with the following structure:
    # dir / *_train.pt --> training data
    # dir / *_val.pt --> validation data
    
    TRAIN_DATA_FILES=glob.glob(dir +"*_train.pt")
    VAL_DATA_FILES=glob.glob(dir+"*_val.pt")
    train_dataset = None
    for TRAIN_DATA_FILE in TRAIN_DATA_FILES:
        with open(TRAIN_DATA_FILE, 'rb') as f:
            if train_dataset is None:
                train_dataset = pk.load(f)
            else:
                train_dataset+=pk.load(f)

    val_dataset = None
    for VAL_DATA_FILE in VAL_DATA_FILES:
        with open(VAL_DATA_FILE, 'rb') as f:
            if val_dataset is None:
                val_dataset = pk.load(f)
            else:
                val_dataset+=pk.load(f)   

    train_data_filtered = [data for data in train_dataset if  (torch.mean(abs(data.dif_y) )>= 0.01)]
    val_data_filtered = [data for data in val_dataset if  (torch.mean(abs(data.dif_y) )>= 0.01)]
    print('###############Total samples in directory################')     
    print('Training Samples: ', len(train_dataset))
    print('Filtered Training Samples: ', len(train_data_filtered))
    print('Validation Samples: ', len(val_dataset))
    print('Filtered Validation Samples: ', len(val_data_filtered))
    print('#########################################################')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=cpus)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=cpus)
    return train_loader, val_loader

def build_dataset_2(batch_size, cpus, file):
    #this function is to use only with the diferent data set size process
    #it asumes one training file pass as "file"
    #and validation files in the directory "./data_norm_flows/"
    
    TRAIN_DATA_FILES=glob.glob(file)
    VAL_DATA_FILES=glob.glob("./data_norm_flows/*_val.pt")
    
    train_dataset = None
    for TRAIN_DATA_FILE in TRAIN_DATA_FILES:
        with open(TRAIN_DATA_FILE, 'rb') as f:
            if train_dataset is None:
                train_dataset = pk.load(f)
            else:
                train_dataset+=pk.load(f)

    val_dataset = None
    for VAL_DATA_FILE in VAL_DATA_FILES:
        with open(VAL_DATA_FILE, 'rb') as f:
            if val_dataset is None:
                val_dataset = pk.load(f)
            else:
                val_dataset+=pk.load(f)   

    print('###############################')            
    print('Training Samples', len(train_dataset))
    print('Validation Samples', len(val_dataset))
    print('###############################')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=cpus)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=cpus)
    return train_loader, val_loader

### graph data set utils###