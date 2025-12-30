import os
import torch

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from graph_models.graph_model import EGNN_network
from graph_models.graph_model import MNF_EGNN_network, gradient_EGNN_network
from graph_models.graph_training_utils import train_model_with_configurable_data, check_if_weights_changed, train_with_curriculum  
from graph_models.callbacks import SmoothedLossMonitor
import argparse
 
#GRADIENT = False
#DATA_DIRECTORY = "./data_sync/datasets/nuevo_4and5_eps01_real/"
#DATA_DIRECTORY = "\home\mdelcastillo\proyectos\EGNN-TopologyControl\data_con_edge_attr"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPUS = int(2*os.cpu_count()/3)
print(f"Using device: {DEVICE}, with {CPUS} CPUs")

CONFIG_GRAD = {
    'optimizer': 'adam',
    'output_dims': [32,32],
    'm_dims': [32,32],
    'update_feats': True,
    'update_coors': True,
    'learning_rate': 0.0005,
    'dropout': 0.0,
    'batch_size': 10,
    'epochs': 500,
    'NormCoors': False,
    'aggr': 'mean',
    'edges': True,
    'mode':'grad',
    "gradient_clip_val": 1.0,
    "gradient_clip_algorithm": "norm"
    }

CONFIG_NO_GRAD = {
    'optimizer': 'adam',
    'output_dims': [8,8],
    'm_dims': [4,4],
    'update_feats': True,
    'update_coors': False,
    'learning_rate': 0.001,
    'dropout': 0.0,
    'batch_size': 20,
    'epochs': 1000,
    'NormCoors': False,
    'aggr': 'mean',
    'edges': True,
    'mode':'mnf',
}

def parse_args():
    """
    Parses command-line arguments to configure data loading and training behavior.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train model with flexible dataset selection")

    parser.add_argument("--use_all", action="store_true",
                        help="Use all available data regardless of number of nodes")
    parser.add_argument("--min_nodes", type=int, default=1,
                        help="Minimum number of nodes (ignored if --use_all is set)")
    parser.add_argument("--max_nodes", type=int, default=20,
                        help="Maximum number of nodes (ignored if --use_all is set)")
    parser.add_argument("--filter", type=float, default=None,
                        help="Optional threshold for filtering samples by mean(abs(dif_y))")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Optional suffix to identify saved model")
    parser.add_argument('--grads', action='store_true', help='Para entrenar el modelo de gradiente')
    parser.add_argument('--data_dir', type=str, help='Directorio de datos para entrenamiento')
    parser.add_argument("--curriculum", nargs="+", type=str, default=None,
                        help="Optional list of node ranges for curriculum learning, e.g. '2-4 5-7 8-10'")
   
    return parser.parse_args()


# -------- MAIN -----------

if __name__ == "__main__":
    args = parse_args()
    DATA_DIRECTORY = args.data_dir #or "./data_sync/datasets/nuevo_4and5_eps01_real/"
    USE_ALL_DATA = args.use_all
    NODE_RANGE = (args.min_nodes, args.max_nodes)
    FILTER_THRESHOLD = args.filter
    if args.grads:
        GRADIENT = True
    else:
        GRADIENT = False
    SUFFIX = args.suffix or ("all" if USE_ALL_DATA else f"{NODE_RANGE[0]}to{NODE_RANGE[1]}")
    print("\n========== Training Configuration ==========")
    print(f"Using ALL data:            {USE_ALL_DATA}")
    print(f"Node range:                {NODE_RANGE if not USE_ALL_DATA else 'IGNORED'}")
    print(f"Filter threshold:          {FILTER_THRESHOLD}")
    print(f"Checkpoint suffix:         {SUFFIX}")
    print(f"Gradient mode:             {GRADIENT}")
    print("===========================================\n")

    config_dict = CONFIG_GRAD if GRADIENT else CONFIG_NO_GRAD
    project_name = "graph_model_Grad" if GRADIENT else "graph_model_NoGrad"
    # if GRADIENT:
    #     model=gradient_EGNN_network(**config_dict)
    #     config_dict['old']= True   
    # else:
    #     model = MNF_EGNN_network(**config_dict)
    #     config_dict['old']=True   
        
    model = EGNN_network(**config_dict)
    
    initial_weights = {name: param.clone() for name, param in model.named_parameters()}

    logger = pl_loggers.WandbLogger(log_model="all", project=project_name, config=config_dict)
    smoothed_cb = SmoothedLossMonitor(window_size=4)
    # Si se especificó un currículo, lo usamos
    if args.curriculum:
        # Parseamos los rangos tipo "2-4 5-7 8-10" → [(2,4),(5,7),(8,10)]
        node_curriculum = []
        for r in args.curriculum:
            try:
                a, b = map(int, r.split("-"))
                node_curriculum.append((a, b))
            except ValueError:
                raise ValueError(f"Invalid curriculum range format: {r}. Expected e.g. '3-5'.")

        train_with_curriculum(
            model=model,
            config=config_dict,
            node_curriculum=node_curriculum,
            logger=logger,
            smoothed_cb=smoothed_cb,
            data_directory=DATA_DIRECTORY,
            project_name=project_name,
            CPUS=CPUS,
            filter_threshold=FILTER_THRESHOLD
        )

    else:
        train_model_with_configurable_data(
            model=model,
            config=config_dict,
            logger=logger,
            smoothed_cb=smoothed_cb,
            node_range=NODE_RANGE,
            suffix=SUFFIX,
            data_directory=DATA_DIRECTORY,
            project_name=project_name,
            CPUS=CPUS,
            use_all_data=USE_ALL_DATA,
            filter_threshold=FILTER_THRESHOLD
        )

    if not check_if_weights_changed(model, initial_weights):
        print("No parameters have been modified.")
