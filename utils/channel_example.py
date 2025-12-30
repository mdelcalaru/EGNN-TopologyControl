from matplotlib import cm
from channel_model import expModel
import numpy as np
import matplotlib.pyplot as plt

def Capacity(x, y, cm):
    rate, var = cm.predict(np.vstack((x, y)))
    return rate[0,1]

def adjacency(x, y, cm):
    return cm.adjacency(np.vstack((x, y)))

x = np.zeros((80,2))
y = np.array([0,0])
x[:,0] = np.linspace(0, 20, 80)
b_fit=0.006346333937904484
# --- distintos conjuntos de parámetros a comparar ---
param_sets = [
    {"Ne": 2, "corte": 1/np.sqrt(b_fit), "power": 4.277633130295451},
    {"Ne": 4, "corte": 1/np.sqrt(b_fit), "power": 4.277633130295451},
]

plt.figure()

for pars in param_sets:
    cm_ind = expModel(**pars)
    l_cap = []
    l_adj = []
    for i in range(x.shape[0]):
        l_cap.append(Capacity(x[i,:], y, cm_ind))
        adj = adjacency(x[i,:], y, cm_ind)
        l_adj.append(adj[0,1])
    # Graficar capacity y/o adjacency; ejemplo solo capacity:
    label = f"Ne={pars['Ne']}, corte={pars['corte']}"
    plt.plot(x[:,0], l_cap, label=label)
    plt.plot(x[:,0], l_adj, '--', label=f"Adj {label}")

plt.xlabel('Distance')
plt.ylabel('Capacity')
plt.grid(True, which='both', alpha=0.3)
plt.legend()
plt.show()
