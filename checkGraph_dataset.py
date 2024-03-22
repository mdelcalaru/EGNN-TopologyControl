import pickle as pk
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import glob
import torch
from graph_dataset_utils import edge_index_gen
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

data_directory="data/gnn_grid/"

TRAIN_DATA_FILES=glob.glob(data_directory +"*_train.pt")
VAL_DATA_FILES=glob.glob(data_directory+"*_val.pt")

for VAL_DATA_FILE in VAL_DATA_FILES:
    with open(VAL_DATA_FILE, 'rb') as f:
        val_dataset = pk.load(f)

for TRAIN_DATA_FILE in TRAIN_DATA_FILES:
    with open(TRAIN_DATA_FILE, 'rb') as f:
        train_dataset = pk.load(f)
xs=[]
ys=[]
c_value=[]
for data in train_dataset:
    positions = data.pos
    x = data.x
    edge_index = data.edge_index
    y=data.y
    indxTask=torch.where(x[:,0]==1)
    x_task_config=positions[indxTask]
    indxComm=torch.where(x[:,1]==1)
    x_comm_config=positions[indxComm]

    xs.append(x_comm_config[0][0].item())
    ys.append(x_comm_config[0][1].item())
    c_value.append(y.item())

size_grid=int(np.sqrt(len(c_value)))

print(size_grid)
c_map=np.zeros((size_grid,size_grid))

for i in range(size_grid):
    for j in range(size_grid):
        c_map[i][j]=c_value[i*size_grid+j]


                           
fig =plt.figure(figsize=(12,5))
# Plot the matrix with the origin at the bottom left
#fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
ax0 = fig.add_subplot(1, 1, 1)

divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.05)
im= ax0.imshow(c_map, cmap=plt.get_cmap('viridis'),extent=[xs[0], xs[-1], ys[-1], ys[0]], vmax=c_map.max(), vmin=c_map.min())

plt.colorbar(im, cax=cax)


ax0.set_title('EGNN')
ax0.plot(x_task_config[:,0],x_task_config[:,1],'o', color='red', markersize=15)


fig.suptitle('graph_model')


plt.show()                                   