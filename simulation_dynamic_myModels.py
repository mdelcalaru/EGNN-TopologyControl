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
#from statistical_allAlgorithms import G
from utils.channel_model import expModel
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from graph_dataset_utils import points_to_data
import pickle
from torch_geometric.data import Data
from graph_model import LightningEGNN_net, grad_simulate_stepY, LightningGrad_net, grad_simulate_stepGrad
from utils.utils import evaluar_grilla, evalModelConvexSim, human_readable_duration
from utils.connectivity_optimization import ConnectivityOpt


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''Load Model y'''
artifact_file_y="./model/EGNN_best_model.ckpt"
model_y_file = Path(artifact_file_y)
model_y =LightningEGNN_net.load_from_checkpoint(model_y_file)
print(model_y)

'''Load Model grad'''
artifact_name_grad="model-lp2vl234:v9" #iie-sc/grad_Net_5/model-lp2vl234:v9 avid-oath-49
artifact_dir_grad="./artifacts/" + artifact_name_grad +"/model.ckpt"
model_file_grad = Path(artifact_dir_grad)

if not model_file_grad.exists():
    import wandb
    run = wandb.init()
    artifact = run.use_artifact("iie-sc/grad_Net_5/"+artifact_name_grad , type='model')
    artifact_dir = artifact.download()     

model_grad =LightningGrad_net.load_from_checkpoint(model_file_grad)

# Configuración inicial
canal=expModel(indicatrix=True)
dist=(canal.rango)*1.2
num_free_agents = 3
num_reconfig_agents = 1
space_size = dist

# Inicializar posiciones de los agentes
free_agents_pos = np.random.rand(num_free_agents, 2) * space_size
reconfig_agents_pos = np.random.rand(num_reconfig_agents, 2) * space_size

rep_experimento=10
pasos_experimento = 50
pasos_ajuste = 5
estadistica={}
t0=time.time()

for experiment in range(rep_experimento):
    t1=time.time()
    free_hist=[free_agents_pos]
    Yreconfig_hist=[reconfig_agents_pos]
    Gradreconfig_hist=[reconfig_agents_pos]
    max_c = []
    c_configY = []
    c_configG=[]
    for iter in range(pasos_experimento):
        deltax = (np.random.rand(num_free_agents) - 0.5)*2
        deltay = (np.random.rand(num_free_agents) - 0.5)*2
        for i in range (num_free_agents):
            if free_agents_pos[i, 0] + deltax[i] < 0 or free_agents_pos[i, 0] + deltax[i] > space_size:
                deltax[i] = -deltax[i] 
            free_agents_pos[i, 0] += deltax[i]
            if free_agents_pos[i, 1] + deltay[i] < 0 or free_agents_pos[i, 1] + deltay[i] > space_size:
                deltay[i] = -deltay[i]
            free_agents_pos[i, 1] += deltay[i]
        
        # Reconfigure agents using our model using y gradient
        tE=time.time()
        Yreconfig_hist.append(Yreconfig_hist[-1])
        for i in range(pasos_ajuste):
            grad =grad_simulate_stepY(NA=reconfig_agents_pos, model=model_y, TA=torch.from_numpy(free_agents_pos), device=device,canal=canal, lr=0.3)
            Yreconfig_hist[iter+1] = Yreconfig_hist[iter]+grad.cpu().numpy()

        print(f"Ours y {experiment} en {time.time()-tE}")
        
        # Reconfigure agents using our model direct gradient
        tG=time.time()
        Gradreconfig_hist.append(Gradreconfig_hist[-1])
        for i in range(pasos_ajuste):
            grad =grad_simulate_stepGrad(NA=reconfig_agents_pos, model=model_y, TA=torch.from_numpy(free_agents_pos), device=device,canal=canal, lr=0.3)
            Gradreconfig_hist[iter+1] = Gradreconfig_hist[iter]+grad.cpu().numpy()
        
        print(f"Ours grad {experiment} en {time.time()-tG}")
        
        '''Log experiment data'''
        max_c.append(evaluar_grilla(task_config=free_agents_pos))      
        c_configY.append(evalModelConvexSim(TA=torch.from_numpy(free_agents_pos), NA=torch.from_numpy(Yreconfig_hist[-1])))
        c_configG.append(evalModelConvexSim(TA=torch.from_numpy(free_agents_pos), NA=torch.from_numpy(Gradreconfig_hist[-1])))
        free_hist.append(free_agents_pos) 
        #reconfig_hist.append(reconfig_agents_pos)
        #Gradreconfig_hist.append(L2reconfig_agents_pos)    
        experiment_data={'free_hist':free_hist,'Yreconfig_hist':Yreconfig_hist,'Gradreconfig_hist':Gradreconfig_hist, 'max_c':max_c, 'c_configY': c_configY, 'c_configG':c_configG}
        estadistica[experiment]=experiment_data
        print(f"Experimento {experiment} iteracion {iter}")
    
    print(f"Experimento {experiment} en {human_readable_duration(time.time()-t1)}")

print(f"Total en {time.time()-t0} segundos") 

file=f'experimento_trueValues_{num_free_agents}T{num_reconfig_agents}N_{rep_experimento}rep_{pasos_ajuste}pasos.pkl'

with open(file, 'wb') as f: 
    pickle.dump(estadistica, f)

# Plot results
fig2 = plt.figure()
plt.plot(max_c, label='Max C',linewidth=10)
plt.plot(c_configY, label='C - grad Y', linewidth=10)
plt.plot(c_configG, label='C - Grad',linewidth=10)

plt.xlabel('Iteration')
plt.ylabel('Configuration Socre')
plt.legend()
plt.show()
	