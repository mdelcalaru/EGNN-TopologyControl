import torch
from graph_dataset_generator.graph_builder import points_to_graph
import numpy as np
from utils.channel_model import expModel
from utils.graph_multyflow_max import MNF_graph_solver
import logging

def grad_simulate_stepY(NA, model, TA, device, canal, y_value=False):
    NAx=NA.detach().cpu()
    NAx.requires_grad=True
    data=points_to_graph(TA,NAx,canal, edge_weight=True).to(device)
    xt, edge_index, edge_attr, positions, batch= data.x, data.edge_index, data.edge_attr, data.pos, data.batch
    f_x=model.forward(xt, edge_index, edge_attr,batch, positions)
    f_x.backward()
    grad_f_x=NAx.grad
    if y_value:
        return grad_f_x, f_x.cpu().detach().numpy()
    else:
        return grad_f_x

def grad_simulate_stepGrad(NA, model, TA, device, canal):
    data=points_to_graph(TA,NA,canal, edge_weight=True).to(device)
    grad_f_x=model.evaluate(data)  
    gradNA=grad_f_x[-NA.shape[0]:,:]
    return gradNA#, f_x.cpu().detach().numpy()

def optimize_NA_Ymodel(NA, model, TA, device, max_iter=1000, lr=0.9,beta=0.9, lr_min=1e-5, lr_max=0.9):
    dxy=torch.ones_like(NA)
    iter=0
    NA_history=[]
    NA_history.append(NA)
    v=0
    # Simulation loop
    while iter < max_iter:
        dxy=grad_simulate_stepY(NA_history[-1], model=model, TA=TA, device=device)
        NA_history.append(NA_history[-1].cpu()+dxy*lr)
        
        grad_norm = torch.sum(torch.linalg.norm(dxy))
        if grad_norm < 1e-2:  # criterio de parada
            break
        
        iter+=1
        # Momentum
        v = beta * v + (1 - beta) * dxy
        
        #lr_adj = lr / (grad_norm + 1e-8)
        #lr_adj = torch.clamp(lr_adj, lr_min, lr_max)
        # Actualizamos NA
        NA_next = NA_history[-1].cpu() + lr.cpu() * v.cpu()
        NA_history.append(NA_next)
        
        if grad_norm < 1e-1:
            lr *= 0.9 
            
    return NA_history[-1], iter

def optimize_NA_Gmodel(NA, model, TA, device, max_iter=1000, lr=0.9, beta=0.9, lr_min=1e-5, lr_max=0.9):
    dxy=torch.ones_like(NA)
    iter=0
    NA_history=[]
    NA_history.append(NA)
    v=0
    # Simulation loop
    while iter < max_iter:
        
        dxy=grad_simulate_stepGrad(NA_history[-1], model=model, TA=TA, device=device)
        
        #if iter%100==0:
            #if lr>0.01:
        #    lr=lr*0.95
        grad_norm = torch.sum(torch.linalg.norm(dxy))
        if grad_norm < 1e-2:  # criterio de parada
            break
        
        iter+=1
        # Momentum
        v = beta * v + (1 - beta) * dxy
        
        lr_adj = lr / (grad_norm + 1e-8)
        lr_adj = torch.clamp(lr_adj, lr_min, lr_max)
        # Actualizamos NA
        NA_next = NA_history[-1].cpu() + lr_adj.cpu() * v.cpu()
        NA_history.append(NA_next)
        
        if grad_norm < 1e-1:
            lr *= 0.9 
            
            
    return NA_history[-1], iter

def evaluar_grilla_model(task_config,model,device, canal):
    #canal=expModel(indicatrix=True)
    dist=(canal.rango)*1.0
    rango=0.5
    x=np.linspace(rango,(dist-rango),int(2*((dist))+1))
    y=np.linspace(rango,(dist-rango),int(2*((dist))+1))
    NA=np.array([[0.5,0.5]])
    c_map=np.empty((len(x),len(y)))
    for c_i, i in enumerate(x):
        for c_j, j in enumerate(y):
            NA[0,0]=i
            NA[0,1]=j
            #print(NA)
            try: 
        
                data=points_to_graph(task_config,torch.from_numpy(NA)).to(device)
                c_map[c_i,c_j]=model.evaluate(data).cpu().detach().numpy()
            except:
                c_map[c_i,c_j]=-4

    return np.max(c_map)

def optimize_NA(NA, model, TA, device,model_type,canal,stop_thresh=1e-4, max_iter=1000, lr=5.0, beta=0.8, lr_min=0.01 ,lr_max=50.0, history=False, thr=0.1):
    """
    Optimiza NA usando gradiente adaptativo con momentum.
    Decide automáticamente qué función de gradiente usar según model.MODE.
    NA_init: tensor inicial
    TA: datos de TA
    model: modelo GNN
    max_iter: número máximo de iteraciones
    lr: tasa de aprendizaje inicial
    beta: factor de momentum
    """
    # Inicialización
    iter = 0
    NA_history = [NA.clone().cpu()]
    v = 0
    dxy = torch.ones_like(NA)

    # Elegir función de gradiente según tipo de modelo
    if model_type == 'grad':
        grad_func = grad_simulate_stepGrad
    else:
        grad_func = grad_simulate_stepY
    
    while iter < max_iter:
        # Calculamos gradiente
        dxy = grad_func(NA_history[-1], model=model, TA=TA, device=device, canal=canal)

        # Norma del gradiente
        grad_norm = torch.norm(dxy)
        if grad_norm < stop_thresh:
            break

        iter += 1

        # Momentum
        v= beta * v + (1 - beta) * dxy.cpu()
        # lr adaptativo basado en la magnitud del gradiente
        lr_adj = lr / (grad_norm.cpu() + 1e-8)  # si gradiente grande, lr pequeño
        # Si lr_adj es float
        lr_adj = max(lr_min, min(lr_adj, lr_max))   
        # Actualizamos NA
        #NA_next = NA_history[-1].cpu() + lr_adj * v
        #NA_history.append(NA_next)
        
        
        # --- 🔸 Control fino de qué nodos se actualizan ---
        grad_per_node = torch.norm(dxy, dim=1)   # norma de cada gradiente individual
        max_grad = grad_per_node.max()
        rel_thresh =thr * max_grad              # actualiza nodos con grad > 10% del máximo
        node_mask = grad_per_node > rel_thresh
        #print(node_mask)
        # Clonamos para no sobreescribir el historial
        NA_next = NA_history[-1].cpu().clone()

        # Solo actualizamos los nodos con gradientes grandes
        NA_next[node_mask.cpu()] += lr_adj * v[node_mask.cpu()]
        # Guardamos el nuevo estado
        NA_history.append(NA_next)
        
    if history:
        return NA_history, iter
    else:
        return NA_history[-1], iter

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_NA_trajectory(NA_history, TA=None, interval=200, save_path=None):
    """
    Muestra cómo evolucionan los puntos NA a lo largo de las iteraciones.

    Parámetros:
    ------------
    NA_history : list of tensors
        Lista con las posiciones de NA en cada iteración (cada tensor de forma [N,2]).
    TA : tensor opcional
        Posiciones de los Task Agents (se muestran fijos).
    interval : int
        Tiempo entre frames (ms).
    save_path : str o None
        Si se especifica, guarda la animación como GIF o MP4.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    # Inicialización de los scatter plots
    scat_NA = ax.scatter([], [], color='tab:blue', label='NA')
    scat_TA = None
    if TA is not None:
        scat_TA = ax.scatter(TA[:, 0].cpu(), TA[:, 1].cpu(), color='tab:red', label='TA', marker='x')

    # Ajustar límites del gráfico
    all_points = torch.cat([NA_history[0]] + ([TA] if TA is not None else []))
    ax.set_xlim(all_points[:, 0].min() - 1, all_points[:, 0].max() + 1)
    ax.set_ylim(all_points[:, 1].min() - 1, all_points[:, 1].max() + 1)
    #ax.legend()
    ax.set_title("Evolución de NA durante la optimización")

    # Función de actualización
    def update(frame):
        NA = NA_history[frame]
        scat_NA.set_offsets(NA.cpu())
        ax.set_title(f"Iteración {frame}/{len(NA_history)-1}")
        return scat_NA,

    ani = FuncAnimation(fig, update, frames=len(NA_history), interval=interval, blit=False)

    if save_path:
        ani.save(save_path, writer='pillow')  # .gif
        print(f"Animación guardada en {save_path}")
    else:
        plt.show()

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyArrowPatch

def plot_static_trajectory(NA_history, TA=None, arrow_every=3, title=None, ax=None, show =True):
    """
    Dibuja las trayectorias de cada NA con flechas indicando el sentido.
    
    Parámetros:
    ------------
    NA_history : list[Tensor]
        Lista de tensores con forma [N, 2] (posiciones de cada NA en cada iteración).
    TA : Tensor opcional
        Posiciones de los Task Agents (se muestran fijos en rojo con 'x').
    arrow_every : int
        Cada cuántos pasos poner una flecha (por estética).
    """
    # Asegurar que todos sean tensores en CPU
    NA_tensors = [torch.as_tensor(na, dtype=torch.float32, device='cpu') for na in NA_history]
    trajs = torch.stack(NA_tensors, dim=0).transpose(0, 1)  # [num_nodes, num_iters, 2]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    for i, traj in enumerate(trajs):
        x, y = traj[:, 0].numpy(), traj[:, 1].numpy()
        ax.plot(x, y, '-o', markersize=2, label=f'NA {i}')

        # Agregar flechas cada "arrow_every" pasos
        for j in range(0, len(x) - 1, arrow_every):
            dx = x[j+1] - x[j]
            dy = y[j+1] - y[j]
            arrow = FancyArrowPatch(
                (x[j], y[j]),
                (x[j+1], y[j+1]),
                arrowstyle='->',
                mutation_scale=8,
                color=ax.lines[-1].get_color(),
                linewidth=1.2
            )
            ax.add_patch(arrow)

    if TA is not None:
        ax.scatter(TA[:, 0].cpu(), TA[:, 1].cpu(), color='red', marker='x', s=80, label='TA')
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Trayectoria de cada NA con dirección de movimiento")
    
    #ax.legend()
    ax.axis('equal')
    if show:
        plt.show()
