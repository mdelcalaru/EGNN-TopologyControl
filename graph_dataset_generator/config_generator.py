import torch
import math
import numpy as np
from scipy.spatial.distance import pdist, squareform
from utils.utils import human_readable_duration, console_width_str
from graph_dataset_generator.graph_builder import points_to_graph
from utils.graph_multyflow_max import MNF_graph_solver
from torch_geometric.utils import to_networkx
import networkx as nx
from scipy.spatial import Delaunay

### Configuration generators ###  
# def x_circle(r, n):
#     """
#     Generates n agent positions arranged randomly along a circle of radius r in 2D.

#     Args:
#         r (float): Radius of the circle.
#         n (int): Number of agents.

#     Returns:
#         torch.Tensor: Tensor of shape (n, 2) with (x, y) positions of agents.
#     """
#     angles = 2 * torch.pi * torch.rand(n)  # n random angles between 0 and 2π
#     x = r * torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
#     return x

def x_task_gen(r, n):
    """
    Generates random 2D positions for n task agents within communication radius r.
  
    The first two agents are initialized at (0, 0). The remaining agents are randomly
    placed in the first quadrant, with distances between 0.2*r and r, and angles between π/6 and π/2.
   
    Args:
        r (float): Communication radius.
        n (int): Number of task agents.
        rndm_seed (int, optional): Seed for reproducibility.
    
    Returns:
       torch.Tensor: Positions of shape (n, 2)
    """
    x_task = torch.zeros((n, 2))

    if n >= 2:
        num_to_generate = n - 2

        angles = torch.rand(num_to_generate) * (math.pi/2 - math.pi/6) + math.pi/6
        radii = torch.rand(num_to_generate)* (0.4*r) + 0.6* r

        x_task[2:, 0] = radii * torch.cos(angles)
        x_task[2:, 1] = radii * torch.sin(angles)

    # Special placement for agent 1 along x-axis
    x_task[1, 0] = r * (torch.rand(1) * 0.5 + 0.5)
    return x_task


def x_comm_gen(r, n, task_agents, rndm_seed=None):
    """
    Generates random 2D positions for n communication agents arranged elliptically
    around the bounding box center of given task agents positions.
   
    Args:
        r (float): Communication max radius (used para escala, aunque no se usa explícito aquí).
        n (int): Number of communication agents.
        task_agents (torch.Tensor): Tensor (m, 2) con posiciones de task agents.
        rndm_seed (int, optional): Seed para reproducibilidad.
  
    Returns:
        torch.Tensor: Posiciones (n, 2) de communication agents.
    """
    if rndm_seed is not None:
        torch.manual_seed(rndm_seed)
   
    max_task_agents = torch.max(task_agents, dim=0).values
    min_task_agents = torch.min(task_agents, dim=0).values
    dist_max = max_task_agents - min_task_agents
    centro = min_task_agents + dist_max / 2.0
    
    x_comm = torch.zeros((n, 2))
    # 360 ángulos uniformemente distribuidos
    theta = torch.linspace(0, 2 * math.pi, 360)
    
    for i in range(n):
        # Ángulo aleatorio de 0 a 359
        idx = torch.randint(0, 360, (1,)).item()
        phi = theta[idx]
       
        x = dist_max[0] * torch.cos(phi) * 0.3 + centro[0]
        y = dist_max[1] * torch.sin(phi) * 0.3 + centro[1]
        
        x_comm[i, 0] = x
        x_comm[i, 1] = y
   
    return x_comm

def poisson_disk_sampling(d, D, n_points, width=100, height=100, k=30):
    """
    Generates Poisson Disk Sampled points where each point is at least 'd' apart
    and within 'D' distance from at least one other point.

    Parameters:
        d (float): Minimum distance between points
        D (float): Maximum distance between points
        n_points (int): Desired number of points
        width, height (int): Bounding box dimensions
        k (int): Number of attempts per active point before removal

    Returns:
        List of (x, y) tuples representing valid points.
    """
    
    cell_size = d / np.sqrt(2)  # Grid cell size for neighbor lookups
    grid_width = int(width / cell_size) + 1
    grid_height = int(height / cell_size) + 1
    
    grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]
    points = []
    active_list = []
    
    def get_grid_coords(x, y):
        """Convert world coordinates to grid indices."""
        return int(x / cell_size), int(y / cell_size)
    
    def is_valid(x, y):
        """Check if the candidate point is at least 'd' away and within 'D' of some point."""
        gx, gy = get_grid_coords(x, y)
        has_nearby = False

        for dx in range(-2, 3):  # Check surrounding grid cells
            for dy in range(-2, 3):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < grid_width and 0 <= ny < grid_height and grid[nx][ny] is not None:
                    px, py = grid[nx][ny]
                    dist = np.hypot(x - px, y - py)
                    if dist < d:  # Too close
                        return False
                    if dist <= D:  # At least one point within D
                        has_nearby = True

        return has_nearby  # Valid only if at least one neighbor is within D
    
    # Step 1: Place the first random point
    first_x, first_y = np.random.uniform(0, width), np.random.uniform(0, height)
    points.append((first_x, first_y))
    active_list.append((first_x, first_y))
    
    gx, gy = get_grid_coords(first_x, first_y)  # FIXED GRID INDEXING
    grid[gx][gy] = (first_x, first_y)
    
    # Step 2: Generate new points until reaching n_points
    while active_list and len(points) < n_points:
        idx = np.random.randint(len(active_list))  # Pick a random active point
        base_x, base_y = active_list[idx]
        found_new_point = False
        
        for _ in range(k):  # Try k candidates around base point
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(d, D)
            new_x = base_x + radius * np.cos(angle)
            new_y = base_y + radius * np.sin(angle)
            
            if 0 <= new_x < width and 0 <= new_y < height and is_valid(new_x, new_y):
                # Accept the new point
                points.append((new_x, new_y))
                active_list.append((new_x, new_y))
                
                gx, gy = get_grid_coords(new_x, new_y)  # FIXED GRID INDEXING
                grid[gx][gy] = (new_x, new_y)
                
                found_new_point = True
                break  # Stop searching if a valid point is found
        
        if not found_new_point:
            active_list.pop(idx)  # Remove point from active list if no valid neighbors
    
    return points

def sort_delaunay_config(params, puntos=None):
    T=params['task_agents']
    N=params['comm_agents']
    #rango=params['channel'].rango
    lado=params['space_size']#params['distance_factor']* rango * np.sqrt(T)

    # Paso 1: puntos iniciales en [0, 1] x [0, 1]
    #densidad=1/rango**2
    #lado=params['distance_factor']*rango*T**0.5

    if T < 3:
        raise ValueError(f"Not enough Task agents ({T}) to sort Delaunay.")
    if puntos is None:
        puntos = np.random.rand(T, 2)*lado
    else:
        puntos = np.array(puntos)
        
    for _ in range(int(N)):
        
        tri = Delaunay(puntos)
        triangles = puntos[tri.simplices]  # coordenadas de los triángulos

        # Calcular área de cada triángulo (usando producto cruzado)
        a = triangles[:, 0]
        b = triangles[:, 1]
        c = triangles[:, 2]
        area = 0.5 * np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
            (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )

        # Encontrar el triángulo más grande
        idx = np.argmax(area)
        tri_grande = triangles[idx]

        # Centroide del triángulo más grande
        centroide = np.mean(tri_grande, axis=0)
        puntos = np.vstack([puntos, centroide])

    task_config=torch.from_numpy(puntos[:T])
    comm_config=torch.from_numpy(puntos[T:])
    return task_config, comm_config
    
def config_from_poisson_disk_sampling(params):
    """
    Genera una configuración de agentes utilizando muestreo tipo Poisson disk, 
    separando los agentes en dos grupos: task y comm.

    Los agentes 'task' se seleccionan entre los generados como los más alejados entre sí,
    maximizando su distancia mínima mutua.

    Parámetros:
    -----------
    params : dict
        Debe incluir:
            - 'channel'.rango : distancia de comunicación.
            - 'total_agents' : cantidad total de agentes.
            - 'task_agents' : cantidad de agentes del tipo task.

    Retorna:
    --------
    task_config : torch.Tensor
        Coordenadas (x, y) de los agentes task (shape: [task_agents, 2]).

    comm_config : torch.Tensor
        Coordenadas (x, y) de los agentes comm restantes (shape: [total_agents - task_agents, 2]).
    """
    rango = params['channel'].rango
    d, D = rango / 3, rango * params['distance_factor']
    n_points = params['total_agents']
    
    #maxsize = params['distance_factor']* rango * np.sqrt(n_points)
    maxsize=params['space_size']
    width = 2*maxsize# * np.random.rand() + maxsize / 2
    height = 2*maxsize#* np.random.rand() + maxsize / 2

    agents_positions = poisson_disk_sampling(d=d,D= D, n_points=n_points, width=width, height=height)
   
    agents_positions_tensor = torch.tensor(agents_positions)
    dist_matrix = squareform(pdist(agents_positions))  # matriz de distancias

    num_task = params['task_agents']
    selected_indices = []

    # Elegimos el primer punto arbitrariamente
    selected_indices.append(0)

    # Iterativamente elegimos el punto que tenga la máxima distancia mínima 
    # a los puntos ya seleccionados, hasta completar la cantidad deseada
    
    for _ in range(1, num_task):
        remaining_indices = list(set(range(n_points)) - set(selected_indices))
        min_distances = []

        for idx in remaining_indices:
            dist_to_selected = dist_matrix[idx, selected_indices]
            min_distances.append(np.min(dist_to_selected))

        next_idx = remaining_indices[np.argmax(min_distances)]
        selected_indices.append(next_idx)

    selected_indices = torch.tensor(selected_indices)

    mask = torch.zeros(n_points, dtype=torch.bool)
    mask[selected_indices] = True

    task_config = agents_positions_tensor[mask]
    comm_config = agents_positions_tensor[~mask]

    return task_config, comm_config

def sort_feasible_config(params, dist='poisson'):
    """
    Generates a feasible configuration of task and communication agents for large teams.

    The function repeatedly samples agent positions (using Poisson disk sampling) and 
    evaluates their connectivity and minimum flow rate (MFR) until a configuration 
    satisfies feasibility conditions (i.e., connected graph and an optimal flow solution).

    If 'task_agents' and 'comm_agents' are not provided in `params`, the function selects 
    the number of task agents randomly in each iteration.

    Parameters:
    -----------
    params : dict
        Dictionary containing:
            - 'total_agents': total number of agents to place.
            - 'task_agents': (optional) number of task agents.
            - 'comm_agents': (optional) number of communication agents.
            - 'channel': object providing `.adjacency(config)` and `.predict(config)` methods.
            - Other fields used by `config_from_poisson_disk_sampling`.

    Returns:
    --------
    task_config : torch.Tensor
        Coordinates of task agents. Shape: [T, 2].

    comm_config : torch.Tensor
        Coordinates of communication agents. Shape: [N, 2].

    MFR : float
        Minimum Flow Rate computed by the MNF solver.
    """
    import logging  
    logger = logging.getLogger(__name__)
    
    status = 'infeasible'

    while status != 'optimal':
        # If not set externally, choose task/comm agents
        if 'task_agents' not in params or 'comm_agents' not in params:
            task_agents = torch.randint(
                low=int(params['total_agents'] / 2),
                high=int(2 * params['total_agents'] / 3),
                size=(1,)
            ).item()
            comm_agents = params['total_agents'] - task_agents
            params['task_agents'] = task_agents
            params['comm_agents'] = comm_agents
        else:
            params['total_agents'] = params['task_agents'] + params['comm_agents']
    
        params['rango'] = params['channel'].rango
        #params['space_size'] = params['distance_factor'] * params['rango'] * ((params['task_agents']/3)**0.5)    
        #print('la distribucion elegida es: ', dist)    
        if dist == 'poisson':
            # Sample configuration using Poisson disk sampling
            task_config, comm_config = config_from_poisson_disk_sampling(params)
        elif dist == 'delaunay':
            logger.info("Generating configuration using Delaunay triangulation...")
            # Sample configuration using Delaunay triangulation
            task_config, comm_config = sort_delaunay_config(params)
        elif dist == 'custom':
            # Sample configuration using custom task agent generator
            radio=params['space_size']
            task_config = x_task_gen(radio, params['task_agents'])
            comm_config = x_comm_gen(radio, params['comm_agents'], task_config)
        else:
            raise ValueError(f"Unknown distribution method: {dist}")
        
        if params['task_agents']<5:
            if torch.cdist(task_config,task_config).max()<params['space_size']*0.6:
                status='too small'
            
        # Stack configurations to form full agent position matrix
        config = np.vstack((task_config.numpy(), comm_config.numpy()))
        # Compute adjacency and channel rates
        adj = params['channel'].adjacency(config)
        rate, _ = params['channel'].predict(config)

        # Build graph from positions
        data_ini = points_to_graph(TA=task_config, NA=comm_config, canal=params['channel'], edge_weight=True)
        G = to_networkx(data_ini, to_undirected=True)

        # Check connectivity and compute MNF if connected
        if nx.is_connected(G):
            try:
                MNF, status = MNF_graph_solver(params['task_agents'], params['comm_agents'], adj, rate)
            except Exception:
                status = 'infeasible'
        else:
            status = 'infeasible'

    return task_config, comm_config, MNF

def shift_config(task_config, comm_config, epsilon=0.1):
    """
    Randomly shifts the positions of task and communication agents.

    Each agent's position is independently perturbed by a small random value 
    sampled uniformly from [-epsilon/2, epsilon/2] along both x and y axes.

    This can be used to introduce slight variations to an existing configuration.

    Parameters:
    -----------
    task_config : torch.Tensor
        Tensor of shape [T, 2] containing the (x, y) positions of task agents.

    comm_config : torch.Tensor
        Tensor of shape [N, 2] containing the (x, y) positions of communication agents.

    epsilon : float, optional (default=0.1)
        Maximum total range of displacement for each coordinate (±epsilon/2).

    Returns:
    --------
    task_config_sub : torch.Tensor
        Shifted positions of task agents.

    comm_config_sub : torch.Tensor
        Shifted positions of communication agents.
    """
    task_config_sub = epsilon * (torch.rand(task_config.shape) - 0.5) + task_config
    comm_config_sub = epsilon * (torch.rand(comm_config.shape) - 0.5) + comm_config

    return task_config_sub, comm_config_sub

def gradient_group_generator(task_config, comm_config, params):
    """
    Generates a list of locally shifted, feasible graph data samples based on 
    initial task and communication agent positions.
    """
    def is_valid_config(TA, NA):
        """Check if the configuration is connected and has an optimal MNF solution."""
        config_np = np.vstack((TA.numpy(), NA.numpy()))
        adj = params['channel'].adjacency(config_np)
        rate, _ = params['channel'].predict(config_np)
        data = points_to_graph(TA=TA, NA=NA, canal=params['channel'], edge_weight=True)
        G = to_networkx(data, to_undirected=True)
        if not nx.is_connected(G):
            return None, 'infeasible'
        try:
            MNF, status = MNF_graph_solver(params['task_agents'], params['comm_agents'], adj, rate)
            if status == 'optimal':
                data_with_MNF = points_to_graph(TA=TA, NA=NA, canal=params['channel'], MNF=MNF, edge_weight=True)
                return data_with_MNF, status
        except:
            pass
        return None, 'infeasible'

    local_data_list = []

    while len(local_data_list) < params['local_sample_count']:
        shifted_TA, shifted_NA = shift_config(task_config, comm_config, params['epsilon'])
        shifted_data, status = is_valid_config(shifted_TA, shifted_NA)
        if status == 'optimal':
            local_data_list.append(shifted_data)

    diff_pos = torch.zeros((params['task_agents']+params['comm_agents'],2,params['local_sample_count'],params['local_sample_count']))
    diff_y = torch.zeros((params['local_sample_count'],params['local_sample_count']))
    for j in range(params['local_sample_count']):
        for k in range(j+1,params['local_sample_count']):
            diff_pos[:,:,j,k]=local_data_list[k].pos-local_data_list[j].pos
            diff_pos[:,:,k,j]=local_data_list[j].pos-local_data_list[k].pos
            diff_y[j,k]=local_data_list[k].y-local_data_list[j].y
            diff_y[k,j]=local_data_list[j].y-local_data_list[k].y

    for l, data in enumerate(local_data_list):
        data.dif_pos=diff_pos[:,:,l,:]#.reshape(1,params['local_sample_count'],params['task_agents']+params['comm_agents'],2)
        data.dif_y=diff_y[l,:].reshape(1,params['local_sample_count'])

    return local_data_list
