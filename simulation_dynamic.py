from calendar import c
import time
from networkx import difference
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pathlib import Path
from math import ceil
from pyparsing import line
import torch
from torch.utils.data import DataLoader
from utils.channel_model import expModel
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from graph_dataset_utils import points_to_data
import pickle
from torch_geometric.data import Data
from graph_model import LightningEGNN_net, grad_simulate_stepY
from utils.utils import evaluar_grilla, evalModelConvexSim, human_readable_duration
from utils.connectivity_optimization import ConnectivityOpt


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# '''Load Model'''
# artifact_file="./model/EGNN_best_model.ckpt"
# model_file = Path(artifact_file)
# model =LightningEGNN_net.load_from_checkpoint(model_file)
# #print(model)
'''Load Model'''

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
# Configuración inicial
canal=expModel(indicatrix=True)
dist=(canal.rango)*1.5
num_free_agents = 3
num_reconfig_agents = 1
space_size = dist

# Inicializar posiciones de los agentes
free_agents_pos = np.random.rand(num_free_agents, 2) * space_size
reconfig_agents_pos = np.random.rand(num_reconfig_agents, 2) * space_size

rep=1
pasos_ajuste = 2
max_c = []
c_config = []
estadistica={}
t0=time.time()

for experiment in range(rep):
    t1=time.time()
    free_hist=[free_agents_pos]
    reconfig_hist=[reconfig_agents_pos]
    L2reconfig_hist=[reconfig_agents_pos]
    max_c = []
    c_config = []
    c_L2=[]
    for iter in range(50):
        deltax = (np.random.rand(num_free_agents) - 0.5)*2
        deltay = (np.random.rand(num_free_agents) - 0.5)*2
        for i in range (num_free_agents):
            if free_agents_pos[i, 0] + deltax[i] < 0 or free_agents_pos[i, 0] + deltax[i] > space_size:
                deltax[i] = -deltax[i] 
            free_agents_pos[i, 0] += deltax[i]
            if free_agents_pos[i, 1] + deltay[i] < 0 or free_agents_pos[i, 1] + deltay[i] > space_size:
                deltay[i] = -deltay[i]
            free_agents_pos[i, 1] += deltay[i]
        tL=time.time()
        # Initialize and run Max L2 optimization
        conOpt=ConnectivityOpt(channel_model=canal, x_task=free_agents_pos, x_comm=L2reconfig_hist[-1])
        copt_config=conOpt.maximize_connectivity(viz=False)
        L2reconfig_agents_pos=copt_config[-num_reconfig_agents:,:]
        print(f"L2 {experiment} en {time.time()-tL}")
        # Reconfigure agents using our model
        tE=time.time()
        for i in range(pasos_ajuste):
            grad =grad_simulate_stepY(NA=reconfig_agents_pos, model=model, TA=torch.from_numpy(free_agents_pos), device=device,canal=canal, lr=0.9)
            reconfig_agents_pos += grad.numpy()

        print(f"Ours {experiment} en {time.time()-tE}")
        '''Log experiment data'''
        max_c.append(evaluar_grilla(task_config=free_agents_pos))      
        c_config.append(evalModelConvexSim(TA=torch.from_numpy(free_agents_pos), NA=torch.from_numpy(reconfig_agents_pos)))
        c_L2.append(evalModelConvexSim(TA=torch.from_numpy(free_agents_pos), NA=torch.from_numpy(L2reconfig_agents_pos)))    
        free_hist.append(free_agents_pos) 
        reconfig_hist.append(reconfig_agents_pos)
        L2reconfig_hist.append(L2reconfig_agents_pos)    
        experiment_data={'free_hist':free_hist,'reconfig_hist':reconfig_hist,'L2reconfig':L2reconfig_hist, 'max_c':max_c, 'c_config': c_config, 'c_L2':c_L2}
        estadistica[experiment]=experiment_data
        print(f"Experimento {experiment} iteracion {iter}")
    
    print(f"Experimento {experiment} en {human_readable_duration(time.time()-t1)}")

print(f"Total en {time.time()-t0} segundos") 

file=f'experimento_sinNorm_{num_free_agents}T{num_reconfig_agents}N_{rep}rep_{pasos_ajuste}pasos.pkl'

with open(file, 'wb') as f: 
    pickle.dump(estadistica, f)

# Plot results
fig2 = plt.figure()
plt.plot(max_c, label='Max C',linewidth=10)
plt.plot(c_L2, label='C - L2',linewidth=10)
plt.plot(c_config, label='C Config', linewidth=10)
plt.xlabel('Iteration')
plt.ylabel('Configuration Socre')
plt.legend()
plt.show()
	