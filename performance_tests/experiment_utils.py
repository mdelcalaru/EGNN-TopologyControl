import torch
import pickle
import time
from pathlib import Path
from tqdm import tqdm
from graph_dataset_generator.config_generator import sort_feasible_config, sort_delaunay_config
from test_utils.evaluation_utils import evalModelConvex
from test_utils.graph_net_utils import grad_simulate_stepY, grad_simulate_stepGrad, optimize_NA
from test_utils.trajectories import gen_task_agents_path
from utils.connectivity_optimization import ConnectivityOpt
from utils.utils import human_readable_duration
import logging
from utils.MNF_NoSA import Simulation
import numpy as np
import copy
def evolve_task_agents(params, task_config, D):
    """Evolves the task agents' positions randomly whitin a windows."""
    #deltax = ((torch.rand(params['task_agents']) - 0.5) * 2).to(task_config.device)
    #deltay = ((torch.rand(params['task_agents']) - 0.5) * 2).to(task_config.device)
    #deltapos=0.1*((task_config)/torch.norm(task_config,dim=1,keepdim=True))
    #deltax=0.5*((task_config[:,0]-task_config[0,0])/torch.norm(task_config,dim=1))
    #deltay=0.5*((task_config[:,1]-task_config[0,1])/torch.norm(task_config,dim=1))
    # = (task_config[:, 0] + deltax < 0) | (task_config[:, 0] + deltax > params['space_size']*2.0)
    #mask_y = (task_config[:, 1] + deltay < 0) | (task_config[:, 1] + deltay > params['space_size']*2.0)
    #deltax[mask_x] *= -1
    #deltay[mask_y] *= -1
    #task_config[:, 0] += deltax
    #task_config[:, 1] += deltay
    #theta=2.0 * (np.pi/180.0)
    #R=torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], device=task_config.device)
    #deltas = torch.matmul(task_config, R.T) - task_config
    #return task_config@R.T+ 2.0*(torch.rand((params['task_agents'], 2), device=task_config.device))
    #return task_config + 2.0*(torch.rand(params['task_agents'], 2, device=task_config.device))
    return task_config + 1.5*D+2.0*(torch.rand(params['task_agents'], 2, device=task_config.device))


def run_dynamic_experiment(params, models_dict,ex):
    """
    Versión que evalúa todos los modelos en el mismo set de experimentos.
    models_dict:
    {
        "graph": {"modelY": model_y_graph, "modelG": model_grad_graph},
        "small": {"modelY": model_y_small, "modelG": model_grad_small},
        ...
    }
    """
    device=params['device']
    logger = logging.getLogger()
    canal = params['channel']
    experiment_steps = params['experiment_steps']
    aproximations_steps = params['aproximations_steps']
    dist=params['dist'] #inicial distribution of agents
    task_config, comm_config, MNF_ini = sort_feasible_config(params, dist=dist)
    params['task_agents'] = task_config.shape[0]
    params['comm_agents'] = comm_config.shape[0]

    msg = (f"Experimento {ex+1}: {params['task_agents']} agentes de tarea, "
            f"{params['comm_agents']} agentes de comunicación: MNF_ini={MNF_ini}")
    tqdm.write(msg)
    logger.info(msg)
    step_size = params['step_size']  # tamaño de paso para evolución dinámica
    # Inicializar historial por modelo
    models_data = {
        model_name: {
            'modelY': {'comm_config_hist': [comm_config.clone()], 'MNF_hist': [MNF_ini], 'time_hist': [], 'y_evol': []},
            'modelG': {'comm_config_hist': [comm_config.clone()], 'MNF_hist': [MNF_ini], 'time_hist': []}
        }
        for model_name in models_dict
    }

    # Free history (solo 1 porque es igual para todos)
    free_hist = [task_config.clone()]
    #noSA data
    noSA_config_hist = [comm_config.clone()]    
    mnf_noSA_hist = [MNF_ini]
    noSA_time_hist = []
    eval_interval=1
    tope_L2=100
    thetas = np.random.uniform(0, 2*np.pi, size=params['task_agents'])
    D = np.column_stack((np.cos(thetas), np.sin(thetas)))  # (N, 2)

    #maxL2 data
    if params['total_agents']>tope_L2:
        msg = f"Experimento {ex+1}: Saltando MaxL2 por muchos agentes ({params['total_agents']})."
        tqdm.write(msg)
        logger.info(msg)
    else:
        maxL2_config_hist = [comm_config.clone()]
        mnf_maxL2_hist = [MNF_ini]
        maxL2_time_hist = []

    for step in tqdm(range(experiment_steps), desc=f'Pasos experimento {ex+1}', position=1):
        task_config = evolve_task_agents(params, task_config,D)
        free_hist.append(task_config.clone())
        # if step%10==0:
        #     thetas = np.random.uniform(0, 2*np.pi, size=params['task_agents'])
        #     D = np.column_stack((np.cos(thetas), np.sin(thetas)))  # (N, 2)

        # --- Evaluar cada modelo ---
        for model_name, model_pair in models_dict.items():
            # Modelo Y
            NAY = models_data[model_name]['modelY']['comm_config_hist'][-1]
            tMNFY = time.time()
            y_evol=[]
            #for _ in range(aproximations_steps):
            NA_Y, iter = optimize_NA(NA=comm_config, model=model_pair['modelY'],model_type ='mnf', TA=task_config, 
                                 stop_thresh=1e-3,device=device, max_iter=3000,canal=canal,thr=0.01)
            #    grad, y_v = grad_simulate_stepY(NA=NAY, model=model_pair['modelY'], TA=task_config, device=device, canal=canal, y_value=True)
                # grad_norm= torch.norm(grad.cpu(),dim=1,keepdim= True)
                # scaling=torch.clamp(grad_norm,min=1e-8)
                # grad_scaled=grad.cpu()*(torch.minimum(grad_norm, torch.tensor(step_size,device=grad_norm.device))/scaling)
                # NAY = NAY + grad_scaled#grad.cpu()*step_size/torch.norm(grad.cpu())
            #    NAY = NAY + grad.cpu()*step_size/torch.norm(grad.cpu(),dim=1, keepdim=True)
            #    y_evol.append(y_v)
            models_data[model_name]['modelY']['time_hist'].append(time.time()-tMNFY)    
            models_data[model_name]['modelY']['comm_config_hist'].append(NAY.clone())
            #models_data[model_name]['modelY']['y_evol'].append(copy.deepcopy(y_evol))
            if step%eval_interval==0:
                models_data[model_name]['modelY']['MNF_hist'].append(evalModelConvex(TA=task_config, NA=NAY, canal=canal))
            else:
                models_data[model_name]['modelY']['MNF_hist'].append(models_data[model_name]['modelY']['MNF_hist'][-1])
            #logger.info(f"Experimento {ex+1} {step} para modelo {model_name} en {models_data[model_name]['modelY']['time_hist'][-1]} segundos")
            # Modelo Grad
            NAG = models_data[model_name]['modelG']['comm_config_hist'][-1]
            tMNFG = time.time()
            #for _ in range(aproximations_steps):
            #    grad = grad_simulate_stepGrad(NA=NAG, model=model_pair['modelG'], TA=task_config, device=device, canal=canal)
                # grad_norm= torch.norm(grad.cpu(),dim=1,keepdim=True)
                # scaling=torch.clamp(grad_norm,min=1e-8)
                # grad_scaled=grad.cpu()*(torch.minimum(grad_norm, torch.tensor(step_size,device=grad_norm.device))/scaling)
                # NAG = NAG + grad_scaled
            #    NAG = NAG + grad.cpu()*step_size/torch.norm(grad.cpu())
            NAG, iter = optimize_NA(NA=comm_config, model=model_pair['modelG'], TA=task_config, 
                                 model_type='grad', device=device, max_iter=3000, canal=canal)
            models_data[model_name]['modelG']['time_hist'].append(time.time()-tMNFG)
            models_data[model_name]['modelG']['comm_config_hist'].append(NAG.clone())
            if step%eval_interval==0:
                models_data[model_name]['modelG']['MNF_hist'].append(evalModelConvex(TA=task_config, NA=NAG, canal=canal))
            else:
                models_data[model_name]['modelG']['MNF_hist'].append(models_data[model_name]['modelG']['MNF_hist'][-1])
            #logger.info(f"Experimento {ex+1} {step} para modelo {model_name} en {models_data[model_name]['modelG']['time_hist'][-1]} segundos")
        if params['total_agents']<=tope_L2:
            # MaxL2 baseline
            conOpt = ConnectivityOpt(channel_model=canal,
                                        x_task=task_config.numpy(),
                                        x_comm=maxL2_config_hist[-1].numpy())
            tMaxL2 = time.time()
            copt_config = conOpt.maximize_connectivity(viz=False, max_its=1, init_step_size=step_size)
            xNA_fin_maxL2 = copt_config[-params['comm_agents']:, :]
            maxL2_time_hist.append(time.time()-tMaxL2)
            maxL2_config_hist.append(torch.from_numpy(xNA_fin_maxL2).clone())
            if step%eval_interval==0:
                mnf_maxL2_hist.append(evalModelConvex(TA=task_config, NA=maxL2_config_hist[-1], canal=canal))
            else:
                mnf_maxL2_hist.append(mnf_maxL2_hist[-1])
#            mnf_maxL2_hist.append(evalModelConvex(TA=task_config, NA=maxL2_config_hist[-1], canal=canal))
            #logger.info(f"Experimento {ex+1} MaxL2 en {maxL2_time_hist[-1]} segundos")
        #noSA baseline
        sim_c_1 = Simulation(d_max= params['rango'], 
                     P0=params['channel'].power, 
                     Ne=params['channel'].Ne, 
                     num_task_agents=params['task_agents'],
                     num_net_agents=params['comm_agents'],
                     TA_pos_0=task_config.cpu().numpy(),
                     NA_pos_0=comm_config.cpu().numpy(), 
                     nmbr_iterations=2, 
                     gradient_ascent_steps=500,
                     alpha_0=step_size)
        tNOSA=time.time()
        xNA_fin_noSA=sim_c_1.simulate()
        noSA_time_hist.append(time.time()-tNOSA)
        noSA_config_hist.append(torch.from_numpy(xNA_fin_noSA).clone())
        if step%eval_interval==0:
            mnf_noSA_hist.append(evalModelConvex(TA=task_config, NA=noSA_config_hist[-1], canal=canal))
        else:
            mnf_noSA_hist.append(mnf_noSA_hist[-1])
#        mnf_noSA_hist.append(evalModelConvex(TA=task_config, NA=noSA_config_hist[-1], canal=canal))
        #logger.info(f"Experimento {ex+1} noSA en {noSA_time_hist[-1]} segundos")

    # Guardar resultados del experimento
    ini_data = {'MNF': MNF_ini, 'task_config': task_config, 'comm_config': comm_config}
    expe_data = {
        'task_agents': params['task_agents'],
        'comm_agents': params['comm_agents'],
        'mode': 'dynamic',
    }
    s = {'free_hist': free_hist,
        'exp': expe_data,
        'ini': ini_data,
        'noSA': {'MNF_hist': mnf_noSA_hist, 'comm_config_hist': noSA_config_hist, 'time_hist': noSA_time_hist},
        'modelos': models_data
    }
    if params['total_agents']<=tope_L2:
        s['maxL2'] = {'MNF_hist': mnf_maxL2_hist, 'comm_config_hist': maxL2_config_hist, 'time_hist': maxL2_time_hist}
        
    return s

def run_static_experiment(params, models_dict,ex):
    """
    Evalúa todos los modelos (en dict) en el mismo set de configuraciones estáticas.
    models_dict:
    {
        "graph": {"modelY": model_y_graph, "modelG": model_grad_graph},
        "small": {"modelY": model_y_small, "modelG": model_grad_small},
        ...
    }
    """
    device=params['device']
    logger = logging.getLogger()
    canal = params['channel']

    dist=params['dist'] #inicial distribution of agents
    task_config, comm_config, MNF_ini = sort_feasible_config(params, dist=dist)
    params['task_agents'] = task_config.shape[0]
    params['comm_agents'] = comm_config.shape[0]
    # print(f'task_agents: {task_config}')
    msg = (f"Experimento {ex+1}: {params['task_agents']} agentes de tarea, "
            f"{params['comm_agents']} agentes de comunicación: MNF_ini={MNF_ini}")
    tqdm.write(msg)
    logger.info(msg)
    
    # Guardar datos del experimento
    ini_data = {
        'MNF': MNF_ini,
        'task_config': task_config,
        'comm_config': comm_config
    }
        
    expe_data = {
        'task_agents': params['task_agents'],
        'comm_agents': params['comm_agents'],
        'mode': 'static'
    }

    s = {
        'exp': expe_data,
        'ini': ini_data,
    }
    # Baseline MaxL2
    if params['total_agents']>40:
        msg = f"Experimento {ex+1}: Saltando MaxL2 por muchos agentes ({params['total_agents']})."
        tqdm.write(msg)
        logger.info(msg)
        convergio = False
    else:
        tM1 = time.time()
        try:
            conOpt = ConnectivityOpt(channel_model=canal,
                                    x_task=task_config.cpu().numpy(),
                                    x_comm=comm_config.cpu().numpy())
            copt_config = conOpt.maximize_connectivity(viz=False)
            xNA_fin_maxL2 = copt_config[-params['comm_agents']:, :]
            tTotal_maxL2 = time.time() - tM1
            convergio = True
            mnf_maxL2 = evalModelConvex(TA=task_config, NA=xNA_fin_maxL2, canal=canal)
            maxL2_data = {
                'MNF': mnf_maxL2,
                'comm_config': xNA_fin_maxL2,
                'convergio': convergio,
                'time': tTotal_maxL2
            }
            s['maxL2'] = maxL2_data
        except Exception as e:
            msg = f"Experimento {ex+1}: Error en maxL2 ({e}). Usando configuración inicial."
            tqdm.write(msg)
            logger.info(msg)
            convergio = False

    tN1 = time.time()
    try:
        # Initialize and run Nacho's simulation
        sim_c_1 = Simulation(d_max=canal.rango, P0=canal.power, Ne=canal.Ne, num_task_agents=params['task_agents'],num_net_agents=params['comm_agents'],TA_pos_0=task_config.cpu().numpy(),NA_pos_0=comm_config.cpu().numpy())
        xNA_fin_noSA=sim_c_1.simulate()
        tTotal_noSA = time.time() - tN1
        mnf_noSA = evalModelConvex(TA=task_config, NA=xNA_fin_noSA,canal=canal)
        conv_noSa = True
        noSA_data = {
            'MNF': mnf_noSA,
            'comm_config': xNA_fin_noSA,
            'convergio': True,
            'time': tTotal_noSA
        }
        s['noSA'] = noSA_data
    except Exception as e:
        msg = f"Experimento {ex+1}: Error en noSA's simulation ({e})."
        tqdm.write(msg) 
        logger.info(msg)
        conv_noSa = False
    # Evaluar todos los modelos
    models_data = {}
    for nombre_modelo, modelos in models_dict.items():
        model_result = {}
        # Modelo Y
        tMNFY = time.time()
        NA_Y, iter = optimize_NA(NA=comm_config, model=modelos["modelY"],model_type ='mnf', TA=task_config, 
                                 stop_thresh=1e-3,device=device, max_iter=3000,canal=canal,thr=0.01)
        tTotal_MNFY = time.time() - tMNFY
        mnf_modelY = evalModelConvex(TA=task_config, NA=NA_Y,canal=canal)
        logger.info(f"modelo {nombre_modelo} y en {iter}")
        model_result['modelY'] = {
            'MNF': mnf_modelY,
            'comm_config': NA_Y,
            'time': tTotal_MNFY
        }
        if iter >= 3000:
            msg = f"Experimento {ex+1}: model_Y {nombre_modelo} salio por iteraciones"
            tqdm.write(msg)
            logger.info(msg)
            model_result['modelY']['convergio']=False
        else:
            model_result['modelY']['convergio']=True

        # Modelo G
        tMNFG = time.time()
        NA_G, iter = optimize_NA(NA=comm_config, model=modelos["modelG"], TA=task_config, 
                                 model_type='grad', device=device, max_iter=3000, canal=canal)
        tTotal_MNFG = time.time() - tMNFG
        mnf_modelG = evalModelConvex(TA=task_config, NA=NA_G,canal=canal)
        logger.info(f"modelo {nombre_modelo} g en {iter}")
        model_result['modelG'] = {
            'MNF': mnf_modelG,
            'comm_config': NA_G,
            'time': tTotal_MNFG
        }
        if iter >= 3000:
            msg = f"Experimento {ex+1}: model_G {nombre_modelo} salio por iteraciones"
            tqdm.write(msg)
            logger.info(msg)
            model_result['modelG']['convergio']=False
        else:
            model_result['modelG']['convergio']=True

        models_data[nombre_modelo] = model_result

    s['modelos']= models_data

        
    # estadistica.append(s)
    for nombre_modelo, resultados in models_data.items():
        logger.info(f"Resultados para {nombre_modelo}:")
        for modelo, datos in resultados.items():
            logger.info(
                f"  {modelo}: "
                f"MNF={datos['MNF']:.4f}, "
                f"Tiempo={datos['time']:.2f}s, "
            )   
    logger.info(f"[ini_data] MNF: {ini_data['MNF']:.4f}" )
    if convergio:
        logger.info(f"[maxL2_data] MNF: {maxL2_data['MNF']:.4f}, "
                    f"Time: {maxL2_data['time']:.2f}s")    
    if conv_noSa:
        logger.info(f"[noSA_data] MNF: {noSA_data['MNF']:.4f}, "
                    f"Time: {noSA_data['time']:.2f}s")
    return s #estadistica

def run_trajectories_experiment(params, models_dict,trajetories):
    """
    Versión que evalúa todos los modelos en el mismo set de experimentos.
    models_dict:
    {
        "name1": {"modelY": model_noGrad_graph, "modelG": model_grad_graph},
    "name1": {"modelY": model_noGrad_graph, "modelG": model_grad_graph},
    ...
    }
    """
    device=params['device']
    logger = logging.getLogger()
    canal = params['channel']
    aproximations_steps = params['aproximations_steps']
    step_size = params['step_size']  # tamaño de paso para evolución dinámica
    print(f'configurando trayectorias... con {params["task_agents"]} agentes de tarea y {params["comm_agents"]} de comunicación')
    if trajetories == 'trebol':
        task_agents_path=gen_task_agents_path(task_agents=params['task_agents'], scale=params['space_size'], lobes=max(5, params['task_agents']), v_factor=0.6)
    
    experiment_steps = len(task_agents_path[0])
    params['experiment_steps']=experiment_steps
    task_config =np.array([task_agents_path[i][0] for i in range(params['task_agents'])])
    task_config, comm_config = sort_delaunay_config(params, task_config)
    MNF_ini = evalModelConvex(TA=task_config, NA=comm_config, canal=canal)
    # Inicializar historial por modelo
    models_data = {
        model_name: {
            'modelY': {'comm_config_hist': [comm_config.clone()], 'MNF_hist': [MNF_ini], 'time_hist': []},
            'modelG': {'comm_config_hist': [comm_config.clone()], 'MNF_hist': [MNF_ini], 'time_hist': []}
        }
        for model_name in models_dict
    }

    # Free history (solo 1 porque es igual para todos)
    free_hist = [task_config.clone()]
    maxL2_config_hist = [comm_config.clone()]
    noSA_config_hist = [comm_config.clone()]    
    mnf_maxL2_hist = [MNF_ini]
    mnf_noSA_hist = [MNF_ini]
    maxL2_time_hist = []
    noSA_time_hist = []

    
    for step in tqdm(range(1,experiment_steps), desc=f'Pasos experimento', position=1):
        task_config = torch.from_numpy(np.array([task_agents_path[i][step] for i in range(params['task_agents'])]))
        free_hist.append(task_config.clone())
        # --- Evaluar cada modelo ---
        for model_name, model_pair in models_dict.items():
            # Modelo Y
            NAY = models_data[model_name]['modelY']['comm_config_hist'][-1]
            tMNFY = time.time()
            for _ in range(aproximations_steps):
                grad = grad_simulate_stepY(NA=NAY, model=model_pair['modelY'], TA=task_config, device=device, canal=canal)
                NAY = NAY + grad.cpu()*step_size/torch.norm(grad.cpu())
            models_data[model_name]['modelY']['time_hist'].append(time.time()-tMNFY)    
            models_data[model_name]['modelY']['comm_config_hist'].append(NAY.clone())
            models_data[model_name]['modelY']['MNF_hist'].append(evalModelConvex(TA=task_config, NA=NAY, canal=canal))
            # Modelo Grad
            NAG = models_data[model_name]['modelG']['comm_config_hist'][-1]
            tMNFG = time.time()
            for _ in range(aproximations_steps):
                grad = grad_simulate_stepGrad(NA=NAG, model=model_pair['modelG'], TA=task_config, device=device, canal=canal)
                NAG = NAG + grad.cpu()*step_size/torch.norm(grad.cpu())
            models_data[model_name]['modelG']['time_hist'].append(time.time()-tMNFG)
            models_data[model_name]['modelG']['comm_config_hist'].append(NAG.clone())
            models_data[model_name]['modelG']['MNF_hist'].append(evalModelConvex(TA=task_config, NA=NAG, canal=canal))
        
        # MaxL2 baseline
        conOpt = ConnectivityOpt(channel_model=canal,
                                    x_task=task_config.numpy(),
                                    x_comm=maxL2_config_hist[-1].numpy())
        tMaxL2 = time.time()
        copt_config = conOpt.maximize_connectivity(viz=False, max_its=aproximations_steps, init_step_size=step_size)
        xNA_fin_maxL2 = copt_config[-params['comm_agents']:, :]
        maxL2_time_hist.append(time.time()-tMaxL2)
        maxL2_config_hist.append(torch.from_numpy(xNA_fin_maxL2).clone())
        mnf_maxL2_hist.append(evalModelConvex(TA=task_config, NA=maxL2_config_hist[-1], canal=canal))
        
        #noSA baseline
        sim_c_1 = Simulation(num_task_agents=params['task_agents'],num_net_agents=params['comm_agents'],alpha_0=0.5, decreasing_alpha=False,
                             TA_pos_0=task_config.cpu().numpy(),NA_pos_0=comm_config.cpu().numpy(), gradient_ascent_steps=aproximations_steps)
        tNOSA=time.time()
        xNA_fin_noSA=sim_c_1.simulate()
        noSA_time_hist.append(time.time()-tNOSA)
        noSA_config_hist.append(torch.from_numpy(xNA_fin_noSA).clone())
        mnf_noSA_hist.append(evalModelConvex(TA=task_config, NA=noSA_config_hist[-1], canal=canal))
        #logger.info(f"Experimento {ex+1} noSA en {noSA_time_hist[-1]} segundos")

    # Guardar resultados del experimento
    ini_data = {'MNF': MNF_ini, 'task_config': task_config, 'comm_config': comm_config}
    expe_data = {
        'task_agents': params['task_agents'],
        'comm_agents': params['comm_agents'],
        'mode': 'trajectories',
    }
    s = {'free_hist': free_hist,
        'exp': expe_data,
        'ini': ini_data,
        'maxL2': {'MNF_hist': mnf_maxL2_hist, 'comm_config_hist': maxL2_config_hist, 'time_hist': maxL2_time_hist},
        'noSA': {'MNF_hist': mnf_noSA_hist, 'comm_config_hist': noSA_config_hist, 'time_hist': noSA_time_hist},
        'modelos': models_data
    }
    
    return s

def save_results(params, estadistica, models, ex=None, output_dir="resultados/", mode='dynamic'):
    """
    Guarda resultados de corridas multi-modelo con metadatos extendidos.
    """
    # Asegurar que sean listas de strings (rutas)
    # model_y_files = list(model_y_files)
    # model_grad_files = list(model_grad_files)

    metadata = {
        'mode': mode,
            'models': [
        {'name': name, 'model_y_file': pathY, 'model_grad_file': pathG}
        for name, (pathY, pathG) in models.items()
    ],
        'rango': params['rango'],
        'space_size': params['space_size'],
        'num_experiments': params['num_experiments'],
        'device': str(params['device']),
    }

    if mode == 'dynamic' or mode == 'trebol':
        metadata['aproximations_steps'] = params['aproximations_steps']
        metadata['experiment_steps'] = params['experiment_steps']

    # Crear carpeta si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Guardar .pkl con datos + metadatos
    if mode == 'trebol':
        result_file = Path(output_dir) / f"{mode}_nodos{params['task_agents']+params['comm_agents']}_trebol_multi.pkl"
    else:
        result_file = Path(output_dir) / f"{mode}_nodos{params['total_agents']}_{ex}rep_multi.pkl"
    with open(result_file, 'wb') as f:
        pickle.dump({'metadata': metadata, 'estadistica': estadistica}, f)

    # Guardar resumen en .txt
    #summary_file = Path(output_dir) / f"{mode}_summary_nodos{ex}_multi.txt"
    #with open(summary_file, 'w') as f:
    #    for k, v in metadata.items():
    #        f.write(f"{k}: {v}\n")

    tqdm.write(f"\n✅ Resultados guardados en {result_file}")