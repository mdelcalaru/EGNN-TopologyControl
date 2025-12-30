import copy
import torch
from graph_dataset_generator.graph_builder import points_to_graph
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.graph_multyflow_max import MNF_graph_solver
from utils.channel_model import expModel
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from torch_geometric.data import DataLoader
import networkx as nx
from torch_geometric.utils import to_networkx

#eval configuration in MNF graph implementation
def evalModelConvex(NA,TA, canal, flow=False) -> float:
    if torch.is_tensor(TA):
        TAin = TA.detach().cpu().numpy()
    else:
        TAin = TA
    if torch.is_tensor(NA):
        NAin = NA.detach().cpu().numpy()
    else:
        NAin = NA
        
    K=TAin.shape[0]
    N=NAin.shape[0]
    #canal=expModel(indicatrix=True)
    config=np.vstack((TAin, NAin))
    adj=canal.adjacency(config)
    rate,_=canal.predict(config)
    MNF, aux =MNF_graph_solver(num_task_config=K, num_comm_config=N, adj=adj, rate=rate,return_flows=flow)
    if flow:
        return MNF, aux
    else:
        return MNF

#eval MNF graph implementation on a grid for a given NA
# this function is used to evaluate the MNF graph implementation on a grid of points
def grid_eval(task_config, comm_config, net_id, canal):
    K=task_config.shape[0]
    #canal=expModel(indicatrix=True)
    max_y=max(max(task_config[:,1]),max(comm_config[:,1]))
    max_x=max(max(task_config[:,0]),max(comm_config[:,0]))
    min_y=min(min(task_config[:,1]),min(comm_config[:,1]))
    min_x=min(min(task_config[:,0]),min(comm_config[:,0]))
    rango=0.5
    minT=min(min_x,min_y)
    maxT=max(max_x,max_y)
    x=np.linspace(min_x-rango,(max_x+rango),int(((maxT-minT)//2)+1))
    y=np.linspace(min_y-rango,(max_y+rango),int(((maxT-minT)//2)+1))
    c_map=np.empty((len(x),len(y)))
    NA=copy.deepcopy(comm_config)
    #comm_config.copy()
    for c_i, i in enumerate(x):
        for c_j, j in enumerate(y):
            NA[net_id-K,0]=i
            NA[net_id-K,1]=j
            config=np.vstack((task_config, NA))
            adj=canal.adjacency(config)
            rate,_=canal.predict(config)
            mnf,status=MNF_graph_solver(num_task_config=K, num_comm_config=comm_config.shape[0], adj=adj, rate=rate)
            c_map[c_i,c_j]=mnf
    
    return c_map, x, y
#evaluates max of grid with only 1 NA
def evaluar_max_grilla(task_config, canal):
    K = task_config.shape[0]
    NA = np.array([[0.5, 0.5]])
    c_map, x, y = grid_eval(task_config, NA, K+1,canal)
    return np.max(c_map)

#Numeric evaluation to compare gradients
def gradient_grid_eval_numeric(task_config, comm_config, net_id, canal):
    c_map, x, y = grid_eval(task_config, comm_config, net_id,canal)
    grad_y, grad_x = np.gradient(c_map, y, x)
    return c_map, grad_x, grad_y, x, y


def gradient_grid_eval_centralized(task_config, comm_config, net_id,model,x,y, canal):
    #canal = expModel(indicatrix=True)
    K = task_config.shape[0]
    NA = comm_config.clone().detach()

    X, Y = np.meshgrid(x, y)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)

    grads_U = np.empty((len(x), len(y)))
    grads_V = np.empty((len(x), len(y)))

    for idx, (xi, yi) in enumerate(grid_points):
        NA_i = NA.clone().detach().requires_grad_(True)
        with torch.no_grad():
            NA_i[net_id - K, :] = torch.tensor([xi, yi], device=NA.device)
        
        data = points_to_graph(TA=task_config, NA=NA_i, canal=canal, edge_weight=True).to(model.device)
        xt, edge_index, edge_attr, positions, batch = data.x, data.edge_index, data.edge_attr, data.pos, data.batch
        
        y_model = model(x=xt, edge_index=edge_index, edge_attr=edge_attr, positions=positions, batch=batch)
        y_model.backward()
        
        grad = NA_i.grad[net_id - K]
        grads_U.ravel()[idx] = grad[1].cpu().detach().numpy()
        grads_V.ravel()[idx] = grad[0].cpu().detach().numpy()

    return grads_U.T, grads_V.T

def gradient_grid_eval_dist(task_config, comm_config, net_id, model, x, y, canal):
    #canal = expModel(indicatrix=True)
    K = task_config.shape[0]
    NA = comm_config.clone().detach()
    device = next(model.parameters()).device
    
    X, Y = np.meshgrid(x, y)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)

    # --- Gradientes estimados por el modelo
    gridU_mat = np.zeros((len(x), len(y)))
    gridV_mat = np.zeros((len(x), len(y)))

    for idx, (xi, yi) in enumerate(grid_points):
        NA_i = NA.clone().detach().requires_grad_(True)
        with torch.no_grad():
            NA_i[net_id - K, :] = torch.tensor([xi, yi], device=NA.device)
        
        data = points_to_graph(TA=task_config, NA=NA_i, canal=canal, edge_weight=True).to(model.device)
        xt, edge_index, edge_attr, positions, batch = data.x, data.edge_index, data.edge_attr, data.pos, data.batch
        
        with torch.no_grad():
            grads_model = model(x=xt, edge_index=edge_index, edge_attr=edge_attr, positions=positions, batch=batch)
        
        grad = grads_model[net_id, :]
        gridU_mat.ravel()[idx] = grad[1].cpu().detach().numpy()
        gridV_mat.ravel()[idx] = grad[0].cpu().detach().numpy()

    return gridU_mat.T, gridV_mat.T


def overlapGrads_plot(task_config, comm_config, net_id, c_map, x, y, grid_grads_U, grid_grads_V, title=None, ax=None, legend=True):
    K = task_config.shape[0]
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    
    X, Y = np.meshgrid(x, y)

    im1 = ax.imshow(c_map, cmap='gray', extent=(x.min(), x.max(), y.min(), y.max()), 
                    origin='lower', vmax=c_map.max(), vmin=c_map.min())
    ax.quiver(X, Y, grid_grads_U, grid_grads_V, color='red')

    ax.scatter(task_config[:, 0], task_config[:, 1], marker='*', c='green', label='Tarea')
    #ax.scatter(comm_config[:, 0], comm_config[:, 1], marker='*', c='blue', label='Comunicación')
    #ax.scatter(comm_config[net_id - K, 0], comm_config[net_id - K, 1], marker='*', c='white', s=200, edgecolors='black', label='Agente movido')
    #ax.set_yticklabels([])    
    ax.set_xlabel('x [m]')
    #ax.set_ylabel('y', fontsize=12)
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    ax.set_facecolor('white')
    #divider1 = make_axes_locatable(ax)
    #cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    #plt.colorbar(im1,cax=cax1)
    #ax.tick_params(labelsize=7)

    return im1
#funciones auxiliares para comparar performance
def plot_ak_matrix(num_task_agents, flows_ak, ax=None,norm=True):  
    mapK=[]
    for i in range(num_task_agents):
        for j in range(i+1,num_task_agents):
            mapK.append([i,j])

    for i in range(1, num_task_agents):
        for j in range(i):
            if i!=j:
                mapK.append([i,j])

    ak_matrix=torch.zeros((num_task_agents,num_task_agents))
    for i in range(len(mapK)):
        ak_matrix[mapK[i][0],mapK[i][1]]=flows_ak[i]
    
    num_rows, num_cols = ak_matrix.shape
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    if norm:
        im1= ax.imshow((ak_matrix-ak_matrix.min())/(ak_matrix.max()-ak_matrix.min()), cmap=plt.get_cmap('viridis'),origin='lower',vmax=1, vmin=0)
    else:
        im1= ax.imshow(ak_matrix, cmap=plt.get_cmap('viridis'),origin='lower',vmax=ak_matrix.max(), vmin=ak_matrix.min())
    
    ax.set_xticks(ticks=np.arange(num_cols), labels=np.arange(num_cols))
    ax.set_yticks(ticks=np.arange(num_rows), labels=np.arange(num_rows))
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im1, cax=cax1)

def get_rate_matrix(task_config, comm_config,canal, rates):
    data=points_to_graph(TA=task_config,NA=comm_config,canal=canal)
    T=task_config.shape[0]+comm_config.shape[0]
    rates_matrix=torch.zeros((T,T))
    for k in range(len(rates)):
        rates_matrix[data.edge_index[0,k],data.edge_index[1,k]]=rates[k]
    
    return rates_matrix

def plot_rates_matrix(task_config, comm_config,canal, rates, ax=None, norm=True): 
    
    rates_matrix=get_rate_matrix(task_config, comm_config,canal, rates)    
    num_rows, num_cols = rates_matrix.shape
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    if norm:
        im1= ax.imshow((rates_matrix-rates_matrix.min())/(rates_matrix.max()-rates_matrix.min()), cmap=plt.get_cmap('viridis'),origin='lower',vmax=1, vmin=0)
    else:
        im1= ax.imshow(rates_matrix, cmap=plt.get_cmap('viridis'),origin='lower',vmax=rates_matrix.max(), vmin=rates_matrix.min())
            
    ax.set_xticks(ticks=np.arange(num_cols), labels=np.arange(num_cols))
    ax.set_yticks(ticks=np.arange(num_rows), labels=np.arange(num_rows))
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im1, cax=cax1)



def mostrar_diferencia_direcciones(Gx1, Gy1, Gx2, Gy2,
                                    umbral_alto=80, umbral_bajo=45,
                                    umbral_magnitud=1e-3,
                                    mostrar_mascara=False,
                                    titulo='Diferencia angular entre gradientes (grados)',
                                    ax=None):
    """
    Calcula y muestra la diferencia angular (en grados) entre vectores gradiente de dos métodos.
    Opcionalmente muestra una máscara categórica e ignora zonas de bajo gradiente.

    Parámetros:
    - umbral_alto: ángulo (grados) considerado "muy diferente"
    - umbral_bajo: ángulo considerado "algo diferente"
    - umbral_magnitud: si la norma de ambos gradientes es menor que esto, se ignora
    - mostrar_mascara: si True, muestra máscara categórica en lugar del ángulo continuo
    - ax: objeto matplotlib.Axes para componer el gráfico
    """
    dot = Gx1 * Gx2 + Gy1 * Gy2
    norm1 = np.sqrt(Gx1**2 + Gy1**2)
    norm2 = np.sqrt(Gx2**2 + Gy2**2)

    eps = 1e-8
    cos_theta = dot / (norm1 * norm2 + eps)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_diff_deg = np.degrees(np.arccos(cos_theta))

    # Zonas de baja magnitud → marcar como NaN (para transparencia)
    mask_baja_magnitud = (norm1 < umbral_magnitud) | (norm2 < umbral_magnitud)
    angle_diff_deg[mask_baja_magnitud] = np.nan

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    divider1 = make_axes_locatable(ax)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    if mostrar_mascara:
        # Categorías: 0 = ≈igual, 1 = diferente, 2 = muy diferente, NaN = sin datos
        mascara = np.full_like(angle_diff_deg, fill_value=np.nan)
        mascara[angle_diff_deg <= umbral_bajo] = 0
        mascara[(angle_diff_deg > umbral_bajo) & (angle_diff_deg <= umbral_alto)] = 1
        mascara[angle_diff_deg > umbral_alto] = 2

        cmap = plt.colormaps.get_cmap('viridis')
        im = ax.imshow(mascara, cmap=cmap, origin='lower')
        cbar = plt.colorbar(im,cax=cax1, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(['≈I', 'Dif', 'M dif'])
    else:
        im = ax.imshow(angle_diff_deg, cmap='viridis', origin='lower')
        plt.colorbar(im,cax=cax1, label='Ángulo (grados)')

    ax.set_title(titulo)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.tight_layout()
    return angle_diff_deg
