from pathlib import Path
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
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
from graph_model import LightningEGNN_net

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''Load Model'''

canal=expModel(indicatrix=True)
artifact_file="/model/EGNN_best_model.ckpt"
model_file = Path(artifact_file)
model =LightningEGNN_net.load_from_checkpoint(model_file)
#print(model)

dist=(canal.rango)*1.5
TA=torch.tensor([[0.0,0.0],[dist,0.0],[0.0,dist]])
x=np.linspace(0,(dist),int(2*((dist))+1))
y=np.linspace(0,(dist),int(2*((dist))+1))
task_agents=3
comm_agents=1

with open('cvxpy_examples/c_mapCVXPY.pkl', 'rb') as f:
    c_mapCVXPY = pickle.load(f)

c_map=np.empty((len(x),len(y)))

Xm=[]
Ym=[]
grid_grads_U=[]
grid_grads_V=[]

for c_i, i in enumerate(x):
    for c_j, j in enumerate(y):
        NA=torch.tensor([[i,j]],requires_grad=True)
        positions = torch.vstack((TA,NA))
        types=torch.zeros((task_agents+comm_agents,2))
        types[0:task_agents,0]=1
        types[task_agents:,1]=1
        edge_index = edge_index_gen(task_agents,comm_agents)

        edge_index=to_undirected(edge_index)
        data=Data(x=types, edge_index=edge_index, pos=positions).to(device)
        xt, edge_index, edge_attr, positions, batch= data.x, data.edge_index, data.edge_attr, data.pos, data.batch
        y_model=model.forward(xt, edge_index, edge_attr,positions,batch)
        c_map[c_i,c_j]=y_model[0].item()
        y_model.backward()
        Xm.append(i)
        Ym.append(j)
        grid_grads_U.append(NA.grad[0,0])
        grid_grads_V.append(NA.grad[0,1])

fig =plt.figure(figsize=(10,10))
# Plot the matrix with the origin at the bottom left
#fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
ax0 = fig.add_subplot(1, 1, 1)

im= ax0.imshow(c_map, cmap=plt.get_cmap('gray'),extent=[x[0], x[-1], y[-1], y[0]], vmax=c_map.max(), vmin=c_map.min())

ax0.set_xlabel('x coordinate for nodes', fontsize=15)
ax0.set_ylabel('y coordinate for nodes', fontsize=15)
plt.title(r"EGNN gradients over true $P(\boldsymbol{x})$",fontsize=20)
ax0.plot(TA[:,0],TA[:,1],'*', color='red', markersize=15)


ax0.quiver(Xm,Ym, grid_grads_U, grid_grads_V, color='red')
ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_ylim(-0.3, dist+0.3)
ax0.set_xlim(-0.3, dist+0.3)
ax0.set_facecolor("white")
plt.tight_layout()
	
plt.savefig('heatmap_gradients.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()