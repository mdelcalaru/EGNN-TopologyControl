from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.channel_model import expModel
from test_utils.evaluation_utils import evalModelConvex
from graph_dataset_generator.graph_builder import points_to_graph
from torch_geometric.data import Data
from graph_models.graph_model import EGNN_network
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from test_utils.evaluation_utils import *
from graph_models.graph_model import load_models, load_models2
from graph_dataset_generator.config_generator import config_from_poisson_disk_sampling
output=Path('./data_sync/paper_results/')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {
    "EGNN - 5": ("./data_sync/models_ckp/graph_model_NoGrad_playful-brook-55_model_all.ckpt", 
                 "./data_sync/models_ckp/graph_model_Grad_laced-surf-105_model_all.ckpt")}

result_file='./data_sync/paper_results/prueba_gradientes.pkl'
# --- Cargar datos
a_fit = 4.277633130295451
b_fit = 0.006346333937904484
canal = expModel(power=a_fit, Ne=2, corte=1/np.sqrt(b_fit))
dist = canal.rango * 1.0
TA = torch.tensor([[0.0, 0.0], [dist, 0], [0,dist]])
NA = torch.tensor([[0.8, 0.8]])
params={'rango':canal.rango,
        'task_agents': 3,
        'comm_agents': 1,
        'seed': 1210}

#TA, NA = config_from_poisson_disk_sampling(params)#fixed seed for reproducibility
net_id=3

# Usamos una función de evaluación numérica (por diferencias finitas)
c_mapCVXPY, grad_x, grad_y, x, y = gradient_grid_eval_numeric(task_config=TA, comm_config=NA, net_id=net_id, canal=canal)
resultados={
    'task_config':TA,
    'comm_config':NA,
    'net_id':net_id,
    'c_mapCVXPY': c_mapCVXPY,
    'grad_y': grad_y,
    'grad_x':grad_x,
    'x_eval':x,
    'y_eval':y  
}
for name, (pathY, pathG) in MODELS.items():
    print(f"\n=== Evaluando {name} ===")
    if name.startswith("OM"):
        model_y, model_grad = load_models2(pathY, pathG, device)
    else:
        model_y, model_grad = load_models(pathY, pathG, device)
        
    #MODELS[name] = {'modelY': model_y, 'modelG': model_grad}
    # Usamos la evaluación autograd (con el modelo que devuelve gradientes)
    grads_U_autograd, grads_V_autograd = gradient_grid_eval_centralized(task_config=TA, comm_config=NA, net_id=3, model=model_y, x=x, y=y, canal=canal)
    # Usamos la evaluación con el modelo (cuando el modelo devuelve gradientes directamente)
    grads_U_model, grads_V_model = gradient_grid_eval_dist(task_config=TA, comm_config=NA, net_id=3, model=model_grad, x=x, y=y, canal=canal)
    resultados[name]={
        'gradsUautograd':grads_U_autograd,
        'gradsVautograd':grads_V_autograd,
        'gradsUmodel':grads_U_model,
        'gradsVmodel':grads_V_model
        }
    from matplotlib.gridspec import GridSpec

    # --- Gráficos
    X, Y = np.meshgrid(x, y)

    #fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig = plt.figure(figsize=(9, 3))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.05)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 3])  
    # --- Primer plot: Gradientes verdaderos
    overlapGrads_plot(task_config=TA, comm_config=NA, net_id=3, c_map=c_mapCVXPY, x=x, y=y, grid_grads_U=grad_x, grid_grads_V=grad_y, title="Numerical Gradients", ax=ax0, legend=False)
    # --- Segundo plot: AutoGradientes del modelo
    overlapGrads_plot(task_config=TA, comm_config=NA, net_id=3, c_map=c_mapCVXPY, x=x, y=y, grid_grads_U=grads_U_autograd, grid_grads_V=grads_V_autograd, title="Autograd EGNN - C", ax=ax1, legend=False)
    # --- Tercer plot: Gradientes del modelo
    im1=overlapGrads_plot(task_config=TA, comm_config=NA, net_id=3, c_map=c_mapCVXPY, x=x, y=y, grid_grads_U=grads_U_model, grid_grads_V=grads_V_model, title="EGNN - D", ax=ax2, legend=False)

    #divider1 = make_axes_locatable(axs[2])
    #cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    # Obtener las posiciones de los 3 subplots
    pos0 = ax0.get_position()
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()

    # Calcular la altura mínima común
    min_bottom = min(pos0.y0, pos1.y0, pos2.y0)
    max_top = max(pos0.y1, pos1.y1, pos2.y1)
    ax0.set_ylabel('y [m]', fontsize=8)
    for ax in [ax1, ax2]:
        ax.set_yticks([])
    # Aplicar exactamente el mismo alto a la colorbar
    cax.set_position([cax.get_position().x0, min_bottom,
                    cax.get_position().width, max_top - min_bottom])
    cbar=plt.colorbar(im1, cax=cax)
    
    cbar.ax.tick_params(labelsize=7)   # ← tamaño de labels

    #plt.tight_layout()
    plt.savefig(output/f'heatmap_gradients_{name}.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)

with open(result_file, 'wb') as f:
    pickle.dump(resultados, f)