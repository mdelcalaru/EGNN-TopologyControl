import argparse
from graph_models.graph_model import load_models, load_models2, load_legacy_checkpoint
import torch
from datetime import datetime
from utils.channel_model import expModel
import time
import logging
import os
import numpy as np
from convert_old_checkpoints import convert_checkpoint, load_old_checkpoint  
from performance_tests.experiment_utils import run_dynamic_experiment, run_static_experiment, save_results, run_trajectories_experiment
import random
import numpy as np
import torch
from tqdm import tqdm

def set_seed(seed=42):
    # Python's built-in random module
    random.seed(seed)
    
    # Numpy's random module
    np.random.seed(seed)
    
    # PyTorch seed for CPU
    torch.manual_seed(seed)
    
    # PyTorch seed for all GPU devices (if using CUDA)
    torch.cuda.manual_seed_all(seed)
    
    # Make sure to disable CuDNN's non-deterministic optimizations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==== PARÁMETROS DESDE LÍNEA DE COMANDOS ====
def parse_args():
    parser = argparse.ArgumentParser(description="Ejecuta experimentos con modelos de grafos.")

    parser.add_argument("--models",
    nargs="+",         # cualquier cantidad de valores
    metavar="MODEL_VAL",  # nombre genérico para la ayuda
    help=(
        "Lista de modelos en formato: NAME PATH_Y PATH_G. "
        "Se pueden pasar varios modelos seguidos, por ejemplo:\n"
        "--models OMOD ./path_y.ckpt ./path_grad.ckpt SMALL ./path_y2.ckpt ./path_grad2.ckpt"
    ))
    

    # Parámetros de simulación
    parser.add_argument("--task_agents", type=int, default=3, help="Cantidad de task agents.")
    parser.add_argument("--comm_agents", type=int, default=2, help="Cantidad de comm agents.")
    parser.add_argument("--distance_factor", type=float, default=1.0, help="Factor de distancia.")
    parser.add_argument("--step_size", type=float, default=0.5, help="Tamaño de paso para evolución dinámica.")
    parser.add_argument("--aproximations_steps", type=int, default=5, help="Cantidad de pasos de aproximaciones.")
    parser.add_argument("--seed", type=int, default=None, help="Semilla aleatoria.")
    parser.add_argument("--output", type=str, default="./results/multi", help="Carpeta base para guardar resultados.")
    return parser.parse_args()

# ==== MAIN ====
if __name__ == "__main__":
    args = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_fit = 4.277633130295451
    b_fit = 0.006346333937904484
    # Construcción de params
    params = {
        'channel': expModel(power=a_fit, Ne=2, corte=1/np.sqrt(b_fit)),
        'channel_description': "expModel(power=a_fit, Ne=2, corte=1/np.sqrt(b_fit))",
        'ab':[a_fit,b_fit],
        #'channel': expModel(),
        'task_agents': args.task_agents,
        'comm_agents': args.comm_agents,
        'distance_factor': args.distance_factor,
        'aproximations_steps': args.aproximations_steps,        
        'device': DEVICE,
        'step_size': args.step_size,
        'num_experiments': 1
        }
    if args.seed is not None:
        #params['seed']=args.seed
        set_seed(args.seed)
    else:
        params['seed'] =random.randint(0, 10000)
        set_seed(params['seed'])
            
    params['rango'] = params['channel'].rango
    params['space_size'] = params['distance_factor'] * params['rango'] * ((params['task_agents']/3)**0.5)
    # Carpeta de resultados con timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join(args.output, timestamp)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(OUTPUT_DIR, "run_multi_experiment.log"),
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()
    logger.info(f"Parámetros del experimento:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")
        
    # === Procesar modelos ===
    MODELS = {}
    if args.models:
        # args.models es una lista larga [NAME1, pathY1, pathG1, NAME2, pathY2, pathG2, ...]
        if len(args.models) % 3 != 0:
            raise ValueError("La opción --models debe recibir múltiplos de 3 argumentos: NAME PATH_Y PATH_G")
        for i in range(0, len(args.models), 3):
            name, pathY, pathG = args.models[i:i+3]
            logger.info(f"Cargando modelo {name} desde {pathY} y {pathG}")
            if name.startswith("OM"):
                model_y, model_grad = load_models2(pathY, pathG, DEVICE)
            else:
                model_y, model_grad = load_models(pathY, pathG, DEVICE)
            MODELS[name] = {'modelY': model_y, 'modelG': model_grad}
    else:
        raise ValueError("Debe especificar al menos un modelo con --models")
    
    # === Ejecutar experimento ===
    logger.info(f"=== Experimento Trebol ===")
    t0 = time.time()
    results_trebol  = run_trajectories_experiment(params, MODELS, trajetories='trebol')
    save_results(params=params, estadistica=results_trebol, models=MODELS, mode='trebol', output_dir=OUTPUT_DIR)
    t1 = time.time()
    logger.info(f"Tiempo experimento trebol: {t1 - t0:.2f} segundos")

    
    print(f"Resultados guardados en {OUTPUT_DIR}")

