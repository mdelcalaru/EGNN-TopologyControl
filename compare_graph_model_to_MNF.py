from pathlib import Path
from math import ceil
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from utils.channel_model import expModel
import torch
import numpy as np
from utils.utils import evalModelConvexSim
from utils.channel_model import expModel
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from graph_model import LightningEGNN_net
import pickle
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import wandb
from graph_dataset_utils import points_to_data
from matplotlib import gridspec
from matplotlib.ticker import FuncFormatter

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''Load Model'''
artifact_file="./model/EGNN_best_model.ckpt"
model_file = Path(artifact_file)
model =LightningEGNN_net.load_from_checkpoint(model_file)
#print(model)
'''Config experiment'''	
canal=expModel(indicatrix=True)
dist=(canal.rango)*1.0
task_agents=3
comm_agents=1
TA=torch.tensor([[0.0,0.0],[dist,0.0],[0.0,dist]])
NA=torch.tensor([[0.5,0.5]],requires_grad=False)
x=np.linspace(0,(dist),int(2*((dist))+1))
y=np.linspace(0,(dist),int(2*((dist))+1))
#print(x)
file='cvxpy_examples/graph_map_sinNorm.pkl'
with open(file, 'rb') as f:
    c_mapCVXPY = pickle.load(f)

c_map=np.empty((len(x),len(y)))

for c_i, i in enumerate(x):
    for c_j, j in enumerate(y):
        NA[0,0]=i
        NA[0,1]=j
        data = points_to_data(TA=TA, NA=NA,canal=canal).to(device)
        y_model=model.evaluate(data)
        c_map[c_i,c_j]=y_model[0].item()
 
c_map_normalized = (c_map-c_mapCVXPY.min()) / (c_mapCVXPY.max()-c_mapCVXPY.min())
c_mapCVXPY_normalized = (c_mapCVXPY-c_mapCVXPY.min()) / (c_mapCVXPY.max()-c_mapCVXPY.min())
def percentage_formatter(x, pos):
    return f'{x * 100:.0f}%'
                            
fig=plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.])
ax0=plt.subplot(gs[0])
im= ax0.imshow(c_map_normalized, cmap=plt.get_cmap('gray'),origin='lower',extent=[x[0], x[-1], y[0], y[-1]],vmax=1, vmin=0)

ax0.set_title('EGNN', fontsize=17)
ax0.plot(TA[:,0],TA[:,1],'*', color='red', markersize=15)
ax0.set_xlabel('x coordinate for nodes', fontsize=15)
ax0.set_ylabel('y coordinate for nodes', fontsize=15)


ax1 = plt.subplot(gs[1])
im2= ax1.imshow(c_mapCVXPY_normalized, cmap=plt.get_cmap('gray'),origin='lower',extent=[x[0], x[-1], y[0], y[-1]],vmax=1, vmin=0)
cbar=fig.colorbar(im2,cax=ax1.inset_axes([1.05, 0, 0.05, 1]))
cbar.ax.tick_params(labelsize=10)
#cbar.set_ticks([-60,-45,-30,-15])
ax1.plot(TA[:,0],TA[:,1],'*', color='red', markersize=15)
ax1.set_xlabel('x coordinate for nodes', fontsize=15)
#ax1.set_ylabel('y coordinate for nodes')
ax1.set_title('MNF-max', fontsize=17)
fig.suptitle('Heatmap for one network node placement', fontsize=20)

ax0.set_ylim(-0.3, dist+0.3)
ax1.set_ylim(-0.3, dist+0.3)
ax0.set_xlim(-0.3, dist+0.3)
ax1.set_xlim(-0.3, dist+0.3)
ax0.set_facecolor("white")
ax1.set_facecolor("white")

ax0.set_xticks([])
ax0.set_yticks([])
ax1.set_xticks([])
ax1.set_yticks([])
plt.tight_layout()
	
plt.savefig('heatmap.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show() 

                          