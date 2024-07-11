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
from utils.utils import evalModelConvex,plot_config
from utils.connectivity_optimization import ConnectivityOpt
import math
from utils.MNF_NoSA import Simulation
import pickle as pk

num_experiments=50
filename= f"experimentos_paper2/exp3/{num_experiments}Run_"
# Set the floating-point precision
np.set_printoptions(precision=3)
#Set cuda device if available
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Set experiment parameters
#Agents in the configuration
task_agents=8
comm_agents=5
total_agents=task_agents+comm_agents
distance_coeficient=2.5
#Channel definition
canal=expModel(indicatrix=True)
Kopts=np.arange(task_agents*(task_agents-1))

#Load EGNN Model
artifact_name="model-o2aeih18:v19"
artifact_dir="artifacts/" + artifact_name +"/model.ckpt"
model_file = Path(artifact_dir)
model =LightningEGNN_net.load_from_checkpoint(model_file)

# Initialize agents config
dist=(canal.rango)*distance_coeficient


dict_results={'ini_feas':[], 
              'EGNN_feas':[],
              'L2_feas':[],
              'nacho_feas':[],
              'aik_ini':[],
              'aik_EGNN':[], 
              'aik_L2':[], 
              'aik_nacho':[], 
              'C_inicial':[],
              'C_EGNN':[], 
              'C_L2':[], 
              'C_nacho':[]}
#Run the experiment
plot_all=False
experiment=0
while experiment < num_experiments:
    TA = x_task_gen(dist,task_agents)
    #TA=torch.tensor([[0.0,0.0],[0.0,dist],[3*dist/4,dist/2]])

    NAi = x_comm_gen(dist*0.8, comm_agents, task_agents=TA).cpu().numpy()
    #NAi=torch.tensor([[3.0,dist]]).cpu().numpy()

    # Initialize and run Nacho's simulation

    sim_c_1 = Simulation(num_task_agents=task_agents,num_net_agents=comm_agents,TA_pos_0=TA.cpu().numpy(),NA_pos_0=NAi,alpha_0=0.5,beta_2=0.14,beta_1=0.14,l2_penalty=0.5,d_max=canal.corte,seed_ta=78,gradient_ascent_steps=30,nmbr_iterations=500,decreasing_alpha=False)
    xNA_fin_nacho=sim_c_1.simulate()

    # Initialize and run Max L2 optimization
    conOpt=ConnectivityOpt(channel_model=canal, x_task=TA.cpu().numpy(), x_comm=NAi)
    copt_config=conOpt.maximize_connectivity(viz=False)
    xNA_fin_maxL2=copt_config[-comm_agents:,:]

    # Run EGNN optimization
    xNA_fin_EGNN = optimize_NA(NAi, model, TA, device, max_iter=1500, lr=0.3)

    #Verify MFR value for all final configurations:
    ini_feas=True
    EGGN_feas=True
    L2_feas=True
    nacho_feas=True
    try:
        R_inicial, C_inicial, aik_ini=evalModelConvex(NA=NAi, TA=TA)    
        data_ini = points_to_data(TA=TA, NA=torch.from_numpy(NAi)).to(device)
        #data_ini=points_to_data(TA,torch.from_numpy(NAi)).to(device)
        y_model_ini=model.evaluate(data_ini)    
        experiment+=1
        try:
            R_EGNN, C_EGNN, aik_EGNN=evalModelConvex(NA=xNA_fin_EGNN, TA=TA)
        except:
            print('EGNN final configuration not feasable')
            EGGN_feas=False

        data_EGNN = points_to_data(TA=TA, NA=torch.from_numpy(xNA_fin_EGNN)).to(device)
        #data_EGNN=points_to_data(TA,torch.from_numpy(xNA_fin_EGNN)).to(device)
        y_model_EGNN=model.evaluate(data_EGNN)    

        try:
            R_L2, C_L2, aik_L2=evalModelConvex(NA=xNA_fin_maxL2, TA=TA)
        except:
            print('Max L2 final configuration not feasable')
            L2_feas=False
        data_L2 = points_to_data(TA=TA, NA=torch.from_numpy(xNA_fin_maxL2)).to(device)
        #data_L2=points_to_data(TA,torch.from_numpy(xNA_fin_maxL2)).to(device)
        y_model_L2=model.evaluate(data_L2)   

        try:
            R_nacho, C_nacho, aik_nacho=evalModelConvex(NA=xNA_fin_nacho, TA=TA)
        except:
            print('Nacho final configuration not feasable')
            nacho_feas=False

        data_nacho = points_to_data(TA=TA, NA=torch.from_numpy(xNA_fin_nacho)).to(device)
        #data_nacho=points_to_data(TA,torch.from_numpy(xNA_fin_nacho)).to(device)
        y_model_nacho=model.evaluate(data_nacho)   

        print(f"******************* MFR {experiment}************************************")
        print(f"INICIAL  --------->  {C_inicial:.3f}, EGNN: {y_model_ini.item():.3f}")
        print('FINAL:')
        print(f"EGNN  --> MFR_fin:   {C_EGNN:.3f}, EGNN: {y_model_EGNN.item():.3f}")
        print(f"MAXL2 --> MFR_copt:  {C_L2:.3f}, EGNN: {y_model_L2.item():.3f}")
        print(f"NACHO --> MFR_nacho: {C_nacho:.3f}, EGNN: {y_model_nacho.item():.3f}")
        print('**************************************************************')

        dict_results['aik_ini'].append(aik_ini)
        dict_results['aik_EGNN'].append(aik_EGNN)
        dict_results['aik_L2'].append(aik_L2)
        dict_results['aik_nacho'].append(aik_nacho)
        dict_results['C_inicial'].append(C_inicial)
        dict_results['C_EGNN'].append(C_EGNN)
        dict_results['C_L2'].append(C_L2)
        dict_results['C_nacho'].append(C_nacho)
        if plot_all:
            fig2, ax2 = plt.subplots(1,3,figsize=(10,20), sharex=True, sharey=True)
            #fig2.suptitle(f"Team Configuration for experiment {experiment}")
            #fig2.subplots_adjust(left=0.05, bottom=0.06, right=0.95, top=0.94, wspace=0.1)

            #ax2[0].plot(TA[:,0],TA[:,1],'*', color='red', markersize=15)
            #ax2[0].plot(NA_history[0][:,0], NA_history[0][:,1], 'ro')  # Plot points and lines
            #rates_ini=np.sum(R_inicial.reshape(total_agents,total_agents,Kopts.shape[0]),axis=2)
            #plot_config(torch.vstack((TA, torch.from_numpy(NAi))), ax=ax2[0,0], clear_axes=True, show=False, task_ids=np.arange(task_agents),rates=rates_ini)
            #ax2[0,0].set_title(f"Initial - MFR: {C_inicial:.3f}, EGNN: {y_model_ini.item():.3f}")

            #ax2[1].plot(TA[:,0],TA[:,1],'*', color='red', markersize=15)
            #ax2[1].plot(NA_history[-1][:,0], NA_history[-1][:,1], 'ro')  # Plot points and lines
            rates_fin_EGNN=np.sum(R_EGNN.reshape(total_agents,total_agents,Kopts.shape[0]),axis=2)
            plot_config(torch.vstack((TA, torch.from_numpy(xNA_fin_EGNN))), ax=ax2[0], clear_axes=True, show=False, task_ids=np.arange(task_agents),rates=rates_fin_EGNN)
            #ax2[0].set_title(f"EGNN - MNF: {C_EGNN:.3f}",fontsize=20)#, EGNN: {y_model_EGNN.item():.3f}")

            #ax2[2].plot(TA[:,0],TA[:,1],'*', color='red', markersize=15)
            #ax2[2].plot(copt_config[-comm_agents:,0], copt_config[-comm_agents:,1], 'ro')  # Plot points and lines
            rates_L2=np.sum(R_L2.reshape(total_agents,total_agents,Kopts.shape[0]),axis=2)
            plot_config(torch.vstack((TA, torch.from_numpy(xNA_fin_maxL2))), ax=ax2[1], clear_axes=True, show=False, task_ids=np.arange(task_agents),rates=rates_L2)
            #ax2[1].set_title(f"Max L2 - MNF: {C_L2:.3f}",fontsize=20)#, EGNN: {y_model_L2.item():.3f}")

            rates_nacho=np.sum(R_nacho.reshape(total_agents,total_agents,Kopts.shape[0]),axis=2)
            plot_config(torch.vstack((TA, torch.from_numpy(xNA_fin_nacho))), ax=ax2[2], clear_axes=True, show=False, task_ids=np.arange(task_agents),rates=rates_nacho)
            #ax2[2].set_title(f"MFR - no SA - MNF: {C_nacho:.3f}",fontsize=20)#, EGNN: {y_model_nacho.item():.3f}")

            ax2[0].set_facecolor("white")
            ax2[1].set_facecolor("white")
            ax2[2].set_facecolor("white")
            #ax2[1,1].set_facecolor("white")
            #plt.show()
            ax2[0].set_xticks([])
            ax2[0].set_yticks([])
            ax2[1].set_xticks([])
            ax2[1].set_yticks([])
            ax2[2].set_xticks([])
            ax2[2].set_yticks([])
            plt.tight_layout()
            plt.savefig(filename+f"comp_experiment{experiment}.pdf")
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

            plt.savefig(filename+f"boxplot_exp{experiment}.pdf")

    except:
        print('Inicial configuration not feasable')
        ini_feas=False
        continue

aik_dif_nacho =[]
aik_dif_L2    =[]
C_dif_nacho   =[]
C_dif_L2      =[]


for experiment in range(len(dict_results['C_EGNN'])):
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
#print(aik1)
#print(aik2)
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

with open(filename+'task_configs.pt', 'wb') as f:
    pk.dump(TA, f)
with open(filename+'ini_Comm_config.pt', 'wb') as f:
    pk.dump(NAi, f)
with open(filename+'EGNN_Comm_config.pt', 'wb') as f:
    pk.dump(xNA_fin_EGNN, f)
with open(filename+'L2_Comm_config.pt', 'wb') as f:
    pk.dump(xNA_fin_maxL2, f)
with open(filename+'nacho_Comm_config.pt', 'wb') as f:
    pk.dump(xNA_fin_nacho, f)
with open(filename+'boxplotAik.pt', 'wb') as f:
    pk.dump(datosAik, f)
with open(filename+'boxplotC.pt', 'wb') as f:
    pk.dump(datosC, f)
    
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.savefig(filename+f"boxplot_{num_experiments}_samples.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.show()
