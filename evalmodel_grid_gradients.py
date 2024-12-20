from pathlib import Path
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from utils.channel_model import expModel
import torch
import numpy as np
#from utils.utils import evalModelConvex
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from graph_dataset_utils import points_to_data
import pickle
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from graph_model import LightningEGNN_net

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''Load Model'''
artifact_file="./model/EGNN_best_model.ckpt"
model_file = Path(artifact_file)
model =LightningEGNN_net.load_from_checkpoint(model_file)
#print(model)


canal=expModel(indicatrix=True)
dist=(canal.rango)*1.0
TA=torch.tensor([[0.0,0.0],[dist,0.0],[0.0,dist]])
x=np.linspace(0,(dist),int(((dist))+1))
y=np.linspace(0,(dist),int(((dist))+1))
task_agents=3
comm_agents=1

with open('cvxpy_examples/graph_map_sinNorm_smallgrid.pkl', 'rb') as f:
    c_mapCVXPY = pickle.load(f)

c_map=np.empty((len(x),len(y)))

Xm=[]
Ym=[]
grid_grads_U=[]
grid_grads_V=[]

for c_i, i in enumerate(x):
    for c_j, j in enumerate(y):
        NA=torch.tensor([[i,j]],requires_grad=True)
        data = points_to_data(TA=TA, NA=NA, canal=canal).to(device)
        xt, edge_index, edge_attr, positions, batch= data.x, data.edge_index, data.edge_attr, data.pos, data.batch
        y_model=model.forward(xt, edge_index, edge_attr,positions,batch)
        c_map[c_i,c_j]=y_model[0].item()
        print(y_model)
        y_model.backward()
        Xm.append(i)
        Ym.append(j)
        grid_grads_U.append(NA.grad[0,0])
        grid_grads_V.append(NA.grad[0,1])


fig =plt.figure(figsize=(6,6))

# ax1 = fig.add_subplot(1, 1, 1)
# divider = make_axes_locatable(ax1)
# cax1 = divider.append_axes("right", size="5%", pad=0.05)

# im1= ax1.imshow(c_map.T, cmap=plt.get_cmap('gray'),origin='lower',extent=[x[0], x[-1], y[0], y[-1]], vmax=c_mapCVXPY.max(), vmin=c_mapCVXPY.min())
# ax1.set_xlabel('x coordinate for nodes', fontsize=15)
# ax1.set_ylabel('y coordinate for nodes', fontsize=15)
# ax1.set_title(r"EGNN gradients over net $P(\boldsymbol{x})$",fontsize=20)
# ax1.plot(TA[:,0],TA[:,1],'*', color='red', markersize=15)
# fig.colorbar(im1,cax=cax1)

# ax1.quiver(Xm,Ym, grid_grads_U, grid_grads_V, color='red')
# ax1.set_xticks([])
# ax1.set_yticks([])
# ax1.set_ylim(-0.3, dist+0.3)
# ax1.set_xlim(-0.3, dist+0.3)
# ax1.set_facecolor("white")

ax2 = fig.add_subplot(1, 1, 1)
# divider = make_axes_locatable(ax2)
# cax2 = divider.append_axes("right", size="5%", pad=0.05)

im2= ax2.imshow(c_mapCVXPY.T, cmap=plt.get_cmap('gray'),origin='lower',extent=[x[0], x[-1], y[0], y[-1]], vmax=c_mapCVXPY.max(), vmin=c_mapCVXPY.min())
ax2.set_xlabel('x coordinate for nodes', fontsize=15)
ax2.set_ylabel('y coordinate for nodes', fontsize=15)
ax2.set_title(r"EGNN gradients over true $P(\boldsymbol{x})$",fontsize=20)
ax2.plot(TA[:,0],TA[:,1],'*', color='red', markersize=15)
#fig.colorbar(im2,cax=cax2)

ax2.quiver(Xm,Ym, grid_grads_U, grid_grads_V, color='red')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_ylim(-0.3, dist+0.3)
ax2.set_xlim(-0.3, dist+0.3)
ax2.set_facecolor("white")

plt.tight_layout()
	
plt.savefig('heatmap_gradients.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()