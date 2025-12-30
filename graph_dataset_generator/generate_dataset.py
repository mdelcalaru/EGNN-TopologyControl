import argparse
from tqdm import tqdm
from graph_dataset_generator.config_generator import sort_feasible_config, gradient_group_generator
from graph_dataset_generator.graph_builder import points_to_graph
from graph_dataset_generator.generator_utils import save_data_list, generate_filename
from utils.channel_model import expModel
from utils.utils import human_readable_duration
import logging
import os
import time
import json
import logging
from tqdm import tqdm
from datetime import datetime
import numpy as np

def main(args):
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(args.out_dir, exist_ok=True)

    # Configuración del logger con soporte para tqdm
    log_path = os.path.join(args.out_dir, f"{timestamp}_generation.log")
    # Configurar el logger
    logging.basicConfig(filename=log_path,
                        filemode="w",  # "a" si querés acumular
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s"
                        )

    logger = logging.getLogger()

    data_list = []
    a_fit = 4.277633130295451
    b_fit = 0.006346333937904484
    # Inicializar parámetros base
    params = {
        'channel': expModel(power=a_fit, Ne=2, corte=1/np.sqrt(b_fit)),
        #'channel':expModel(),
        'num_samples': args.num_samples,
        'distance_factor': args.distance_factor,
        'channel_description': "expModel(power=a_fit, Ne=2, corte=1/np.sqrt(b_fit))",
        'ab':[a_fit,b_fit]
    }
  
    # Agregar parámetros según los argumentos
    #if args.dist is 'Delaunay':
    params['generation']=args.dist
    #    dist = 'delaunay'
    #else:
    dist = args.dist
    if args.task_agents is not None:
        params['task_agents'] = args.task_agents
    if args.comm_agents is not None:
        params['comm_agents'] = args.comm_agents
    if args.total_agents is not None:
        params['total_agents'] = args.total_agents
    else:
        params['total_agents']=args.task_agents + args.comm_agents    
    if args.epsilon is not None:
        params['epsilon'] = args.epsilon
        
    # Verificación de configuración válida
    if (('task_agents' not in params or 'comm_agents' not in params) and 'total_agents' not in params):
        raise ValueError("Params must contain either both 'task_agents' and 'comm_agents', or 'total_agents'.")

    # Manejo de muestras locales
    local_sample_count = args.local_samples if args.local_samples is not None else 1
    params['local_sample_count'] = local_sample_count

    if local_sample_count > 1:
        if args.num_samples % local_sample_count != 0:
            raise ValueError(f"num_samples ({args.num_samples}) is not divisible by local_samples ({local_sample_count}).")
        params['diferent_samples'] = args.num_samples // local_sample_count
    
    params['rango'] = params['channel'].rango
    params['space_size'] = params['distance_factor'] * params['rango'] * (((params['total_agents'])/3)**0.5)

    # Log de parámetros
    msg="Running dataset generation with parameters:"
    logger.info(msg)
    print(msg)
    for key, value in params.items():
        logger.info(f"  {key}: {value}")
        print(f"  {key}: {value}")

    # Guardar configuración en archivo JSON (excluyendo el objeto canal)
    serializable_params = {k: v for k, v in params.items() if k != 'channel'}
    params_path = os.path.join(args.out_dir, f"{timestamp}_params.json")
    with open(params_path, "w") as f:
        json.dump(serializable_params, f, indent=4)
        
    logger.info(f"Saved parameters to {params_path}")
    print(f"Saved parameters to {params_path}")
    # Iniciar generación
    times = []
    if local_sample_count > 1:
        pbar= tqdm(range(params['diferent_samples']), desc="Generating grouped samples")
        for i in pbar:
            t0 = time.perf_counter()
            task_config, comm_config, MNF = sort_feasible_config(params, dist=dist)
            data_group = gradient_group_generator(task_config, comm_config, params)  # <--- debe ser lista
            data_list.append(data_group)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            pbar.set_description(f"Muestra {i+1}/{params['diferent_samples']}")
            pbar.set_postfix({"tiempo (s)": f"{elapsed:.2f}"})
            logger.info(f"Group {i + 1}/{params['diferent_samples']} generated in {elapsed:.2f} s")
    else:
        pbar = tqdm(range(params['num_samples']), desc="Generating samples")
        for i in pbar:
            t0 = time.perf_counter()
            task_config, comm_config, MNF = sort_feasible_config(params, dist=dist)
            data = points_to_graph(TA=task_config, NA=comm_config, canal=params['channel'], MNF=MNF, edge_weight=True)
            data_list.append(data)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            pbar.set_description(f"Muestra {i+1}/{params['num_samples']}")
            pbar.set_postfix({"tiempo (s)": f"{elapsed:.2f}"})
            logger.info(f"Sample {i + 1}/{params['num_samples']} generated in {elapsed:.2f} s")

    avg_time = sum(times) / len(times)
    logger.info(f"Average generation time: {avg_time:.2f} s")
    logger.info(f"Total generation time: {sum(times):.2f} s")
    print(f"Average generation time: {avg_time:.2f} s")
    print(f"Total generation time: {human_readable_duration(sum(times))}")
    
    # División train/val
    if local_sample_count > 1:
        # data_list: lista de grupos (listas de muestras)
        split_idx = int(len(data_list) * 0.80)# 80% para entrenamiento
        # Dividir en grupos de entrenamiento y validación
        # data_list es una lista de grupos, cada grupo es una lista de muestras
        # Por lo tanto, se divide en dos listas de grupos
        train_groups = data_list[:split_idx]
        val_groups = data_list[split_idx:]

        # Aplanar las listas
        data_train = [sample for group in train_groups for sample in group]
        data_val = [sample for group in val_groups for sample in group]
    else:
        # data_list: lista plana
        split_idx = int(len(data_list) * 0.85)
        data_train = data_list[:split_idx]
        data_val = data_list[split_idx:]

    # Guardar datasets
    train_path = os.path.join(args.out_dir + "_train.pt")
    val_path = os.path.join(args.out_dir + "_val.pt")

    train_path = generate_filename(args.out_dir, params['total_agents'], params['epsilon'], len(data_train), "train",timestamp,dist)
    val_path = generate_filename(args.out_dir, params['total_agents'], params['epsilon'], len(data_val), "val",timestamp,dist)

    save_data_list(data_train, train_path)
    save_data_list(data_val, val_path)

    logger.info(f"Saved training dataset with {len(data_train)} samples to {train_path}")
    logger.info(f"Saved validation dataset with {len(data_val)} samples to {val_path}")
    print(f"Saved training dataset with {len(data_train)} samples to {train_path}")
    print(f"Saved validation dataset with {len(data_val)} samples to {val_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1250)
    parser.add_argument("--task_agents", type=int, default=None)
    parser.add_argument("--comm_agents", type=int, default=None)
    parser.add_argument("--total_agents", type=int, default=None)
    parser.add_argument("--local_samples", type=int, default=None)
    parser.add_argument("--distance_factor", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--dist", type=str)
    parser.add_argument("--out_dir", type=str, default="dataset")
    args = parser.parse_args()
    main(args)
# This code is part of a script that generates a dataset of graph configurations for a multi-agent system.
# It uses various parameters to create task and communication agent configurations, ensuring they are feasible.
# The generated configurations are saved in a specified output directory, and the process is logged for debugging and analysis.