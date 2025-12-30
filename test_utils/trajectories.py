import numpy as np
from math import pi


def trebol(scale, theta, lobes=5, v_factor=0.4, offset_x=0.0, offset_y=0.0):
    
    pts = np.zeros((theta.size,2))
    rad=scale+scale*v_factor*np.cos(lobes*theta)
    pts[:,0] = rad*np.cos(theta)+offset_x
    pts[:,1] = rad*np.sin(theta)+offset_y
    return pts

def path_length(pts):
    """Calcula longitud total del path (m)"""
    dif = np.diff(pts, axis=0)
    seg_len = np.sqrt((dif**2).sum(axis=1))
    return np.sum(seg_len)

def gen_task_agents_path(task_agents, scale, lobes=3, v_factor=0.7):
    dt=1.0
    vel_target=0.3  # m/s
    theta_dense = np.linspace(0, np.pi, 5000)
    trebol_pts = trebol(scale, theta_dense, lobes, v_factor)
    L = path_length(trebol_pts)
    steps =int(np.ceil(L / vel_target*dt))//3
    print(f"Path length: {L:.2f} m, steps: {steps}")
    task_agents_path={} 
    for i in range(task_agents):
        angles = np.linspace (i*(2.0 * pi)/ task_agents , (2.0 * pi  + i*(2.0 * pi)/ task_agents), num=steps)
        trebol_pts = trebol(scale, angles, lobes, v_factor)
        task_agents_path[i] = trebol_pts
    return task_agents_path
