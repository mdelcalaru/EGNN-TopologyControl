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
from utils.utils import evalModelConvex
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from graph_dataset_utils import points_to_data
import pickle
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from graph_model import LightningEGNN_net, grad_simulate_step
from utils.utils import evaluar_grilla, evalModelConvex
from graph_model import evaluar_grilla_model
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''Load Model'''

canal=expModel(indicatrix=True)

#artifact_file="./model/EGNN_best_model.ckpt"
#model_file = Path(artifact_file)

artifact_name="model-giwgzbdx:v9" #cosmic-wind-321
artifact_dir="artifacts/" + artifact_name +"/model.ckpt"
model_file = Path(artifact_dir)


model =LightningEGNN_net.load_from_checkpoint(model_file)
#print(model)

dist=(canal.rango)*1.0


# Configuración inicial
num_free_agents = 3
num_reconfig_agents = 1
space_size = dist

# Inicializar posiciones de los agentes
free_agents_pos = np.random.rand(num_free_agents, 2) * space_size
reconfig_agents_pos = np.random.rand(num_reconfig_agents, 2) * space_size

# Crear la figura
#fig, ax = plt.subplots()
#ax.set_xlim(0, space_size)
#ax.set_ylim(-4, 1)

# Dibujar agentes
#free_agents_scatter = ax.scatter(free_agents_pos[:, 0], free_agents_pos[:, 1], c='blue', label='Free Agents')
#reconfig_agents_scatter = ax.scatter(reconfig_agents_pos[:, 0], reconfig_agents_pos[:, 1], c='red', label='Reconfig Agents')

pasos_ajuste = 5
max_c = []
c_config = []
#def update(frame):
#    global free_agents_pos, reconfig_agents_pos
    # Mover agentes libres aleatoriamente
estadistica={}
t0=time.time()
for experiment in range(20):
    t1=time.time()
    model_y = []
    free_hist=[]
    reconfig_hist=[]
    max_model_y=[]
    max_c = []
    c_config = []
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
        
        # Reconfigurar agentes en función de los agentes libres
        for i in range(pasos_ajuste):
            grad, y =grad_simulate_step(NA=reconfig_agents_pos, model=model, TA=torch.from_numpy(free_agents_pos), device=device, lr=0.3)
            reconfig_agents_pos += grad.numpy()


        #adj=canal.adjacency(torch.vstack((torch.from_numpy(free_agents_pos),torch.from_numpy(reconfig_agents_pos))).numpy())
        #rate, _=canal.predict(torch.vstack((torch.from_numpy(free_agents_pos),torch.from_numpy(reconfig_agents_pos))).numpy())
        #data=points_to_data(TA=torch.from_numpy(free_agents_pos),NA=torch.from_numpy(reconfig_agents_pos), adj=adj, rate=rate).to(device)
        #model_y.append(model.evaluate(data).cpu().detach().numpy())
        max_c.append(evaluar_grilla(task_config=free_agents_pos))
        #max_model_y.append(evaluar_grilla_model(task_config=torch.from_numpy(free_agents_pos),model=model,device=device))
        try:
            c_config.append(evalModelConvex(TA=torch.from_numpy(free_agents_pos), NA=torch.from_numpy(reconfig_agents_pos)))
        except:
            c_config.append(-4.0)
            
        free_hist.append(free_agents_pos)
        reconfig_hist.append(reconfig_agents_pos)
    
        experiment_data={'free_hist':free_hist,'reconfig_hist':reconfig_hist, 'max_c':max_c, 'c_config': c_config}
        estadistica[experiment]=experiment_data
        print(f"Experimento {experiment} iteracion {iter}")
    
    print(f"Experimento {experiment} en {time.time()-t1} segundos")

print(f"Total en {time.time()-t0} segundos") 
# Actualizar las posiciones en el gráfico
#    free_agents_scatter.set_offsets(free_agents_pos)
#    reconfig_agents_scatter.set_offsets(reconfig_agents_pos)
#    return free_agents_scatter, reconfig_agents_scatter
    
#ani = animation.FuncAnimation(fig, update, frames=100, interval=100, blit=True) 
#plt.legend()
#plt.show()
file='experimento_trueValues_3T1N_5rep_5pasos.pkl'
with open(file, 'wb') as f:
    pickle.dump(estadistica, f)

#model_y = np.array(model_y)
'''
fig2 = plt.figure()
plt.plot(max_c, label='Max C',linewidth=10)
plt.plot(model_y[:,0,0], label='Model Y',linewidth=7)
plt.plot(max_model_y, label='Max Model Y',linewidth=5)
plt.plot(c_config, label='C Config', linewidth=3)
plt.xlabel('Iteration')
plt.ylabel('Configuration Socre')
#plt.ylim(-4, 1)
plt.legend()
plt.show()
'''	