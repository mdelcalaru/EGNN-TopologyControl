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
from utils.utils import evalModelConvex
from utils.connectivity_optimization import ConnectivityOpt
from utils.utils import plot_config
import math
from utils.MNF_NoSA import Simulation
import pickle as pk

num_experiments=1
filename= f"ejemplos_dinamicos/{num_experiments}Run_"
# Set the floating-point precision
np.set_printoptions(precision=3)
#Set cuda device if available
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Set experiment parameters
#Agents in the configuration
task_agents=3
comm_agents=1
total_agents=task_agents+comm_agents
distance_coeficient=1.5
#Channel definition
canal=expModel(indicatrix=True)
Kopts=np.arange(task_agents*(task_agents-1))

#Load EGNN Model
artifact_name="model-z6sd5ldw:v9"
artifact_dir="/home/mdelcastillo/proyectos/mfr_learning/artifacts/" + artifact_name +"/model.ckpt"
model_file = Path(artifact_dir)
model =LightningEGNN_net.load_from_checkpoint(model_file)

# Initialize agents config
dist=(canal.rango)*distance_coeficient


dict_results={'ini_feas':[], 
              'EGNN_feas':[],
              'aik_ini':[],
              'aik_EGNN':[], 
              'C_inicial':[],
              'C_EGNN':[]
              }
#Run the experiment
plot_all=False

for experiment in range(num_experiments):
    TA = x_task_gen(dist,task_agents)
    #TA=torch.tensor([[0.0,0.0],[dist,dist]])#,[0.0,dist]])

    NAi = x_comm_gen(dist*0.8, comm_agents, task_agents=TA).cpu().numpy()
    #NAi=torch.tensor([[3.0,dist]]).cpu().numpy()

    # Initialize and run Nacho's simulation

    xNA_fin_EGNN = optimize_NA(NAi, model, TA, device, max_iter=1500, lr=0.3)

    #Verify MFR value for all final configurations:
    ini_feas=True
    EGGN_feas=True
    L2_feas=True
    nacho_feas=True
    try:
        R_inicial, _ , C_inicial, aik_ini, _=evalModelConvex(NA=NAi, TA=TA.numpy(), channel=canal, Kopts=Kopts)    
        data_ini=points_to_data(TA,torch.from_numpy(NAi)).to(device)
        y_model_ini=model.evaluate(data_ini)    

        try:
            R_EGNN, _, C_EGNN, aik_EGNN, _=evalModelConvex(NA=xNA_fin_EGNN, TA=TA.numpy(), channel=canal, Kopts=Kopts)
        except:
            print('EGNN final configuration not feasable')
            EGGN_feas=False
        data_EGNN=points_to_data(TA,torch.from_numpy(xNA_fin_EGNN)).to(device)
        y_model_EGNN=model.evaluate(data_EGNN)    
 

        print(f"******************* MFR {experiment}************************************")
        print(f"INICIAL  --------->  {C_inicial:.3f}, EGNN: {y_model_ini.item():.3f}")
        print('FINAL:')
        print(f"EGNN  --> MFR_fin:   {C_EGNN:.3f}, EGNN: {y_model_EGNN.item():.3f}")
        print('**************************************************************')

        dict_results['aik_ini'].append(aik_ini)
        dict_results['aik_EGNN'].append(aik_EGNN)
        dict_results['C_inicial'].append(C_inicial)
        dict_results['C_EGNN'].append(C_EGNN)
        
        if plot_all:
            fig2, ax2 = plt.subplots(2,2,figsize=(10,10), sharex=True, sharey=True)
            fig2.suptitle(f"Team Configuration for experiment {experiment}")
            fig2.subplots_adjust(left=0.05, bottom=0.06, right=0.95, top=0.94, wspace=0.1)

            #ax2[0].plot(TA[:,0],TA[:,1],'*', color='red', markersize=15)
            #ax2[0].plot(NA_history[0][:,0], NA_history[0][:,1], 'ro')  # Plot points and lines
            rates_ini=np.sum(R_inicial.reshape(total_agents,total_agents,Kopts.shape[0]),axis=2)
            plot_config(torch.vstack((TA, torch.from_numpy(NAi))), ax=ax2[0,0], clear_axes=True, show=False, task_ids=np.arange(task_agents),rates=rates_ini)
            ax2[0,0].set_title(f"Initial - MFR: {C_inicial:.3f}, EGNN: {y_model_ini.item():.3f}")

            #ax2[1].plot(TA[:,0],TA[:,1],'*', color='red', markersize=15)
            #ax2[1].plot(NA_history[-1][:,0], NA_history[-1][:,1], 'ro')  # Plot points and lines
            rates_fin_EGNN=np.sum(R_EGNN.reshape(total_agents,total_agents,Kopts.shape[0]),axis=2)
            plot_config(torch.vstack((TA, torch.from_numpy(xNA_fin_EGNN))), ax=ax2[0,1], clear_axes=True, show=False, task_ids=np.arange(task_agents),rates=rates_fin_EGNN)
            ax2[0,1].set_title(f"EGNN - MFR: {C_EGNN:.3f}, EGNN: {y_model_EGNN.item():.3f}")

            #ax2[2].plot(TA[:,0],TA[:,1],'*', color='red', markersize=15)
            #ax2[2].plot(copt_config[-comm_agents:,0], copt_config[-comm_agents:,1], 'ro')  # Plot points and lines
            rates_L2=np.sum(R_L2.reshape(total_agents,total_agents,Kopts.shape[0]),axis=2)
            plot_config(torch.vstack((TA, torch.from_numpy(xNA_fin_maxL2))), ax=ax2[1,0], clear_axes=True, show=False, task_ids=np.arange(task_agents),rates=rates_L2)
            ax2[1,0].set_title(f"Max L2 - MFR: {C_L2:.3f}, EGNN: {y_model_L2.item():.3f}")

            rates_nacho=np.sum(R_nacho.reshape(total_agents,total_agents,Kopts.shape[0]),axis=2)
            plot_config(torch.vstack((TA, torch.from_numpy(xNA_fin_nacho))), ax=ax2[1,1], clear_axes=True, show=False, task_ids=np.arange(task_agents),rates=rates_nacho)
            ax2[1,1].set_title(f"Nacho - MFR: {C_nacho:.3f}, EGNN: {y_model_nacho.item():.3f}")

            ax2[0,0].set_facecolor("white")
            ax2[0,1].set_facecolor("white")
            ax2[1,0].set_facecolor("white")
            ax2[1,1].set_facecolor("white")
            #plt.show()

            plt.savefig(f"comp_experiment{experiment}.png")
            #plt.close(fig=fig2)


            fig3, ax3 = plt.subplots(figsize=(8,6))  # Set the figure size

            aik1=aik_ini[aik_ini < 1.0-1e-5]
            aik2=aik_EGNN[aik_EGNN < 1.0-1e-5]
            aik3=aik_L2[aik_L2 < 1.0-1e-5]
            aik4=aik_nacho[aik_nacho < 1.0 - 1e-5]
            print(aik1)
            print(aik2)
            print(aik3)
            print(aik4)
            datos=[aik1, aik2, aik3, aik4]
            etiquetas=['Initial', 'EGNN Gradient descend', 'Max connectivity', 'Nacho' ]
            # Create a box plot
            bplot=ax3.boxplot(datos, patch_artist=True,medianprops={'color': 'darkblue','linewidth': 2})  # Use patch_artist to fill boxes with color

            # Customizations
            ax3.set_facecolor("white")
            #plt.xlabel('version', fontsize=12)  # Set x-axis label and fontsize
            plt.ylabel(r'$a_{i,k}$ Values', fontsize=12)  # Set y-axis label and fontsize
            plt.title(f"Resulted flows for experiment {experiment}", fontsize=14)  # Set title and fontsize
            plt.xticks([1, 2, 3, 4], etiquetas)  # Set custom x-axis tick labels
            plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines with custom linestyle and transparency
            plt.tight_layout()  # Adjust layout to prevent clipping of labels

            # Customize box colors
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow' ]
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

            plt.savefig(f"boxplot_exp{experiment}.png")

    except:
        print('Inicial configuration not feasable')
        ini_feas=False
        continue

aik_dif_nacho =[]
aik_dif_L2    =[]
C_dif_nacho   =[]
C_dif_L2      =[]


for experiment in range(num_experiments):
    aik_dif_nacho.append(100*(dict_results['aik_EGNN'][experiment]-dict_results['aik_nacho'][experiment])/abs(dict_results['aik_EGNN'][experiment]))
    aik_dif_L2.append(100*(dict_results['aik_EGNN'][experiment]-dict_results['aik_L2'][experiment])/abs(dict_results['aik_EGNN'][experiment]))
    C_dif_nacho.append(100*(dict_results['C_EGNN'][experiment]  -dict_results['C_nacho'][experiment])/abs(dict_results['C_EGNN'][experiment]))
    C_dif_L2.append(100*(dict_results['C_EGNN'][experiment]  -dict_results['C_L2'][experiment])/abs(dict_results['C_EGNN'][experiment]))

C_dif_array_L2 = np.array(C_dif_L2)
C_dif_array_nacho = np.array(C_dif_nacho)
aik_dif_array_nacho=np.array(aik_dif_nacho)
aik_dif_array_L2=np.array(aik_dif_L2)

vary=7.5

fig3, ax3 = plt.subplots(1,2,figsize=(6,3))  # Set the figure size

aik1=aik_dif_array_L2[aik_dif_array_L2 < 1000.0]
aik2=aik_dif_array_nacho[aik_dif_array_nacho < 1000.0]
print(aik1)
print(aik2)
datosAik=[aik1, aik2]

#etiquetas=['Initial', 'EGNN Gradient descend', 'Max connectivity', 'Nacho' ]
etiquetas=['L2 Max', 'MFR - no SA' ]
# Create a box plot
bplot=ax3[0].boxplot(datosAik, patch_artist=True,medianprops={'color': 'darkblue','linewidth': 2})  # Use patch_artist to fill boxes with color

# Customizations
ax3[0].set_facecolor("white")
#plt.xlabel('version', fontsize=12)  # Set x-axis label and fontsize
#ax3[0].set_ylabel("Percentual diference in Optimal Flows to our method", fontsize=12)  # Set y-axis label and fontsize
ax3[0].set_title('Optimal flows', fontsize=12)  # Set title and fontsize
ax3[0].set_xticks([1, 2, ], etiquetas)  # Set custom x-axis tick labels
ax3[0].grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines with custom linestyle and transparency
ax3[0].set_ylim(-vary,vary)

# Customize box colors
colors = ['lightblue', 'lightgreen']#, 'lightcoral', 'lightyellow' ]
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)


cl2=C_dif_array_L2[C_dif_array_L2 < 1000.0]
cnacho=C_dif_array_nacho[C_dif_array_nacho < 1000.0]

datosC=[cl2, cnacho]

#etiquetas=['Initial', 'EGNN Gradient descend', 'Max connectivity', 'Nacho' ]
etiquetas=['L2 Max', 'MFR - no SA' ]
# Create a box plot
bplot=ax3[1].boxplot(datosC, patch_artist=True,medianprops={'color': 'darkblue','linewidth': 2})  # Use patch_artist to fill boxes with color

# Customizations
ax3[1].set_facecolor("white")
#plt.xlabel('version', fontsize=12)  # Set x-axis label and fontsize
#ax3[1].set_ylabel("Porcentual difference to our method", fontsize=12)  # Set y-axis label and fontsize
ax3[1].set_title('P values', fontsize=12)  # Set title and fontsize
ax3[1].set_xticks([1, 2, ], etiquetas)  # Set custom x-axis tick labels
ax3[1].grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines with custom linestyle and transparency
ax3[1].set_ylim(-vary,vary)

# Customize box colors
colors = ['lightblue', 'lightgreen']#, 'lightcoral', 'lightyellow' ]
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

with open(filename+'boxplotAik.pt', 'wb') as f:
    pk.dump(datosAik, f)
with open(filename+'boxplotC.pt', 'wb') as f:
    pk.dump(datosC, f)
    
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.savefig(f"boxplot_{num_experiments}_samples.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.show()