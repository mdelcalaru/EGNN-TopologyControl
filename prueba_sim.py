import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pathlib import Path
from math import ceil
import torch
from torch.utils.data import DataLoader
from utils.channel_model import expModel
import torch
import numpy as np
from utils.utils import evalModelConvex
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from graph_dataset_utils import edge_index_gen
import pickle
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from graph_model import LightningEGNN_net, grad_simulate_step

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''Load Model'''

canal=expModel(indicatrix=True)

#artifact_file="./model/EGNN_best_model.ckpt"
#model_file = Path(artifact_file)

artifact_name="model-mlzmc8z4:v9" #cosmic-wind-321
artifact_dir="artifacts/" + artifact_name +"/model.ckpt"
model_file = Path(artifact_dir)


model =LightningEGNN_net.load_from_checkpoint(model_file)
#print(model)

dist=(canal.rango)*1.0


# Configuración inicial
num_free_agents = 3
num_reconfig_agents =2
space_size = dist*2

# Inicializar posiciones de los agentes
free_agents_pos = np.random.rand(num_free_agents, 2) * space_size
reconfig_agents_pos = np.random.rand(num_reconfig_agents, 2) * space_size

# Crear la figura
fig, ax = plt.subplots()
ax.set_xlim(0, space_size)
ax.set_ylim(0, space_size)

# Dibujar agentes
free_agents_scatter = ax.scatter(free_agents_pos[:, 0], free_agents_pos[:, 1], c='blue', label='Free Agents')
reconfig_agents_scatter = ax.scatter(reconfig_agents_pos[:, 0], reconfig_agents_pos[:, 1], c='red', label='Reconfig Agents')

pasos_ajuste = 5

def update(frame, free_agents_pos=free_agents_pos, reconfig_agents_pos=reconfig_agents_pos, num_free_agents=num_free_agents, pasos_ajuste=pasos_ajuste, model=model, device=device):
    # Mover agentes libres aleatoriamente
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
        reconfig_agents_pos += grad_simulate_step(NA=reconfig_agents_pos, TA=torch.from_numpy(free_agents_pos), model=model, device=device, lr=1).numpy()
    
    # Actualizar las posiciones en el gráfico
    free_agents_scatter.set_offsets(free_agents_pos)
    reconfig_agents_scatter.set_offsets(reconfig_agents_pos)
    return free_agents_scatter, reconfig_agents_scatter

ani = animation.FuncAnimation(fig, update, frames=100, interval=100, blit=True)
plt.legend()
plt.show()
