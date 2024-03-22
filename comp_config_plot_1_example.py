import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from math import ceil
import torch
from utils.channel_model import expModel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from graph_model import LightningEGNN_net, optimize_NA
from torch_geometric.data import Data
from graph_dataset_utils import  points_to_data, x_comm_gen, x_task_gen
from utils.utils import evalModelConvex, plot_config
from utils.connectivity_optimization import ConnectivityOpt
import math
from utils.MNF_NoSA import Simulation
import pickle as pk

filename= "run_output/singleRun_"

# Set the floating-point precision
np.set_printoptions(precision=3)
#Set cuda device if available
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Set experiment parameters
#Agents in the configuration
task_agents=3
comm_agents=1
total_agents=task_agents+comm_agents
distance_coeficient=1.0
#Channel definition
canal=expModel(indicatrix=True)
Kopts=np.arange(task_agents*(task_agents-1))

#Load EGNN Model
artifact_file="/model/EGNN_best_model.ckpt"
model_file = Path(artifact_file)
model =LightningEGNN_net.load_from_checkpoint(model_file)

# Initialize agents config
dist=(canal.rango)*distance_coeficient
TA = x_task_gen(dist,task_agents)
NAi = x_comm_gen(dist*0.8, comm_agents, task_agents=TA,rndm_seed=1024).cpu().numpy()


# Initialize and run No Share access simulation

sim_c_1 = Simulation(num_task_agents=task_agents,num_net_agents=comm_agents,TA_pos_0=TA.cpu().numpy(),NA_pos_0=NAi,alpha_0=0.5,beta_2=0.14,beta_1=0.14,l2_penalty=0.5,d_max=canal.corte,seed_ta=78,gradient_ascent_steps=30,nmbr_iterations=500,decreasing_alpha=False)
xNA_fin_noSa=sim_c_1.simulate()

# Initialize and run Max L2 optimization
conOpt=ConnectivityOpt(channel_model=canal, x_task=TA.cpu().numpy(), x_comm=NAi)
copt_config=conOpt.maximize_connectivity(viz=False)
xNA_fin_maxL2=copt_config[-comm_agents:,:]

# Run EGNN optimization
xNA_fin_EGNN = optimize_NA(NAi, model, TA, device, max_iter=2500, lr=0.3)

#Verify MFR value for all final configurations:
ini_feas=True
EGGN_feas=True
L2_feas=True
noSa_feas=True
try:
    R_inicial, _ , C_inicial, aik_ini, _=evalModelConvex(NA=NAi, TA=TA.numpy(), channel=canal, Kopts=Kopts)
except:
    print('Inicial configuration not feasable')
    ini_feas=False

data_ini=points_to_data(TA,torch.from_numpy(NAi)).to(device)
y_model_ini=model.evaluate(data_ini)    

try:
    R_EGNN, _, C_EGNN, aik_EGNN, _=evalModelConvex(NA=xNA_fin_EGNN, TA=TA.numpy(), channel=canal, Kopts=Kopts)
except:
    print('EGNN final configuration not feasable')
    EGGN_feas=False
data_EGNN=points_to_data(TA,torch.from_numpy(xNA_fin_EGNN)).to(device)
y_model_EGNN=model.evaluate(data_EGNN)    

try:
    R_L2, _, C_L2, aik_L2, _=evalModelConvex(NA=xNA_fin_maxL2, TA=TA.numpy(), channel=canal, Kopts=Kopts)
except:
    print('Max L2 final configuration not feasable')
    L2_feas=False
data_L2=points_to_data(TA,torch.from_numpy(xNA_fin_maxL2)).to(device)
y_model_L2=model.evaluate(data_L2)   

try:
    R_noSa, _, C_noSa, aik_noSa, NS=evalModelConvex(NA=xNA_fin_noSa, TA=TA.numpy(), channel=canal, Kopts=Kopts)
except:
    print('NnoSa final configuration not feasable')
    noSa_feas=False


data_noSa=points_to_data(TA,torch.from_numpy(xNA_fin_noSa)).to(device)
y_model_noSa=model.evaluate(data_noSa)   

print(f"******************* MFR *************************************")
if ini_feas:
    print(f"INICIAL  --------->  {C_inicial:.3f}, EGNN: {y_model_ini.item():.3f}")
print('FINAL:')
if EGGN_feas:
    print(f"EGNN  --> MFR_fin:   {C_EGNN:.3f}, EGNN: {y_model_EGNN.item():.3f}")
if L2_feas:
    print(f"MAXL2 --> MFR_copt:  {C_L2:.3f}, EGNN: {y_model_L2.item():.3f}")
if noSa_feas:
    print(f"noSa --> MFR_noSa: {C_noSa:.3f}, EGNN: {y_model_noSa.item():.3f}")
print('**************************************************************')


fig2, (ax0,ax1,ax2) = plt.subplots(1,3,figsize=(12,4), sharex=True, sharey=True)
fig2.suptitle(f"Team Configuration")
fig2.subplots_adjust(left=0.05, bottom=0.06, right=0.95, top=0.94, wspace=0.1)

if EGGN_feas:
    rates_fin_EGNN=np.sum(R_EGNN.reshape(total_agents,total_agents,Kopts.shape[0]),axis=2)
    plot_config(torch.vstack((TA, torch.from_numpy(xNA_fin_EGNN))), ax=ax0, clear_axes=True, show=False, task_ids=np.arange(task_agents),rates=rates_fin_EGNN)
    ax0.set_title(f"EGNN - MFR: {C_EGNN:.3f}, EGNN: {y_model_EGNN.item():.3f}")
    ax0.set_facecolor("white")
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_ylim(-1, dist*1.2)
    ax0.set_xlim(-1, dist*1.2)
    plt.tight_layout()

if L2_feas:
    rates_L2=np.sum(R_L2.reshape(total_agents,total_agents,Kopts.shape[0]),axis=2)
    plot_config(torch.vstack((TA, torch.from_numpy(xNA_fin_maxL2))), ax=ax1, clear_axes=True, show=False, task_ids=np.arange(task_agents),rates=rates_L2)
    ax1.set_title(f"Max L2 - MFR: {C_L2:.3f}, EGNN: {y_model_L2.item():.3f}")
    ax1.set_facecolor("white")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_ylim(-1, dist*1.2)
    ax1.set_xlim(-1, dist*1.2)
    plt.tight_layout()

if noSa_feas:
    rates_noSa=np.sum(R_noSa.reshape(total_agents,total_agents,Kopts.shape[0]),axis=2)
    plot_config(torch.vstack((TA, torch.from_numpy(xNA_fin_noSa))), ax=ax2, clear_axes=True, show=False, task_ids=np.arange(task_agents),rates=noSa)
    ax2[2].set_title(f"noSa- MFR: {C_noSa:.3f}, EGNN: {y_model_noSa.item():.3f}")
    ax2.set_facecolor("white")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_ylim(-1, dist*1.2)
    ax2.set_xlim(-1, dist*1.2)
    plt.tight_layout()


with open(filename+'TA.pt', 'wb') as f:
    pk.dump(TA, f)
with open(filename+'NAi.pt', 'wb') as f:
    pk.dump(NAi, f)
with open(filename+'xNA_fin_EGNN.pt', 'wb') as f:
    pk.dump(xNA_fin_EGNN, f)
with open(filename+'xNA_fin_maxL2.pt', 'wb') as f:
    pk.dump(xNA_fin_maxL2, f)
with open(filename+'xNA_fin_noSa.pt', 'wb') as f:
    pk.dump(xNA_fin_noSa, f)        
with open(filename+'rates_fin_EGNN.pt', 'wb') as f:
    pk.dump(rates_fin_EGNN, f)
with open(filename+'rates_L2.pt', 'wb') as f:
    pk.dump(rates_L2, f)
with open(filename+'rates_noSa.pt', 'wb') as f:
    pk.dump(rates_noSa, f)    

with open(filename+'C.pt', 'wb') as f:
    pk.dump(np.array([C_inicial, C_EGNN, C_L2, C_noSa]), f)
with open(filename + 'y_mpdel.pt', 'wb') as f:
    pk.dump(np.array([y_model_ini.item(), y_model_EGNN.item(), y_model_L2.item(), y_model_noSa.item()]), f)
with open(filename + 'Feas.pt', 'wb') as f:
    pk.dump(np.array([ini_feas, EGGN_feas, L2_feas, noSa_feas]), f)
    
    
plt.savefig(filename+'_config_compare.pdf', format='pdf', bbox_inches='tight', dpi=300)


fig3, ax3 = plt.subplots(figsize=(5,5))  # Set the figure size
datos=[]
etiquetas=[]
ticks=[]
colors = []
if EGGN_feas:
    aik2=aik_EGNN[aik_EGNN < 1.0-1e-5]
    print(aik2)
    datos.append(aik2)
    etiquetas.append('EGNN')
    colors.append('lightblue')
if L2_feas:
    aik3=aik_L2[aik_L2 < 1.0-1e-5]
    datos.append(aik3)
    print(aik3)
    etiquetas.append('Max L2')
    colors.append('lightgreen')
if noSa_feas:
    aik4=aik_noSa[aik_noSa < 1.0 - 1e-5]
    print(aik4)
    datos.append(aik4)
    etiquetas.append('MFR - No SA')
    colors.append('lightcoral')

item=1
for elementos in datos:
    ticks.append(item)
    item+=1

# Create a box plot
bplot=ax3.boxplot(datos, patch_artist=True,medianprops={'color': 'darkblue','linewidth': 2})  # Use patch_artist to fill boxes with color

# Customizations
ax3.set_facecolor("white")
plt.ylabel(r'$a_{i,k}$ Values', fontsize=12)  # Set y-axis label and fontsize
plt.title(f"Resulted flows distribution", fontsize=14)  # Set title and fontsize
plt.xticks(ticks, etiquetas)  # Set custom x-axis tick labels
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines with custom linestyle and transparency
plt.tight_layout()  # Adjust layout to prevent clipping of labels

# Customize box colors
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)


with open(filename+'boxplot_exp_data.pt', 'wb') as f:
    pk.dump(datos, f)
plt.show()


