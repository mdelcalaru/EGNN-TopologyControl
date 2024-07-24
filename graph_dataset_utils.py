import argparse
from pathlib import Path
import datetime
import time
import torch
from utils.utils import human_readable_duration, console_width_str, graph_plot
from utils.MNF_cvxpy import MNF_share_solver
import numpy as np
from utils.channel_model import expModel
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, to_edge_index
import glob
import pickle as pk
from torch_geometric.loader import DataLoader
from utils.graph_multyflow_max import MNF_graph_solver
import math

def build_dataset(batch_size, cpus, dir):
    TRAIN_DATA_FILES=glob.glob(dir +"*_train.pt")
    VAL_DATA_FILES=glob.glob(dir+"*_val.pt")
    train_dataset = None
    for TRAIN_DATA_FILE in TRAIN_DATA_FILES:
        with open(TRAIN_DATA_FILE, 'rb') as f:
            if train_dataset is None:
                train_dataset = pk.load(f)
            else:
                train_dataset+=pk.load(f)

    val_dataset = None
    for VAL_DATA_FILE in VAL_DATA_FILES:
        with open(VAL_DATA_FILE, 'rb') as f:
            if val_dataset is None:
                val_dataset = pk.load(f)
            else:
                val_dataset+=pk.load(f)   

    print('###############################')            
    print('Training Samples', len(train_dataset))
    print('Validation Samples', len(val_dataset))
    print('###############################')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=cpus)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=cpus)
    return train_loader, test_loader


def edge_index_gen(task,comm,adj,rate):
    adjacency_matrix =rate*adj
    edge_index, edge_weight= to_edge_index(adjacency_matrix.to_sparse())
    return edge_index, edge_weight

def points_to_data(TA,NA,mfr=None):
    task_agents=TA.shape[0]
    comm_agents=NA.shape[0]
    positions = torch.vstack((TA,NA))
    canal=expModel(indicatrix=True)
    adj =canal.adjacency(positions.detach().numpy())
    rate, _=canal.predict(positions.detach().numpy())
    types=torch.zeros((task_agents+comm_agents,2))
    types[0:task_agents,0]=1
    types[task_agents:,1]=1
    edge_index, edge_weights = edge_index_gen(task_agents,comm_agents, adj=torch.from_numpy(adj), rate=torch.from_numpy(rate))
    
    edge_index,edge_weights=to_undirected(edge_index,edge_attr=edge_weights)

    if mfr is None:
        data=Data(x=types, edge_index=edge_index,edge_attr=edge_weights, pos=positions)
        #data=Data(x=types, edge_index=edge_index, pos=positions)
    else:
        data=Data(x=types, edge_index=edge_index,edge_attr=edge_weights, pos=positions, y=mfr)
        #data=Data(x=types, edge_index=edge_index, pos=positions, y=mfr)

    return data
  
def x_circle(r, n):
    x_=torch.zeros((n,2))
    theta = torch.linspace(0, 2*np.pi, 360)
    for i in range(n):
        phi=theta[np.random.randint(0, 360-1, 1)][0]
        x_[i,0] = r * torch.cos(phi)
        x_[i,1] = r * torch.sin(phi)
   
    return x_ 

def x_task_gen(r,n, rndm_seed=None):
    rng = np.random.default_rng(seed=rndm_seed)
    x_task=torch.zeros((n,2))
    if n >= 2:
        for i in range(2,n):
            angulo=torch.tensor(rng.uniform(low=torch.pi/6,high=torch.pi/2,size=None))
            radio=r*rng.uniform(low=0.2,high=1.0,size=None)    
            x_task[i,0] = radio*torch.cos(angulo)
            x_task[i,1] = radio*torch.sin(angulo)
    
        x_task[1,0] = r * rng.uniform(low=0.4,high=1.0,size=None) 
    return x_task

def x_comm_gen(r,n, task_agents, rndm_seed=None):
    rng = np.random.default_rng(seed=rndm_seed)
    max_task_agents= torch.max(task_agents, dim=0).values.tolist()
    min_task_agents= torch.min(task_agents, dim=0).values.tolist()
    dist_max=np.array([max_task_agents[0]-min_task_agents[0],max_task_agents[1]-min_task_agents[1]])
    centro=torch.tensor([min_task_agents[0]+dist_max[0]/2,(min_task_agents[1]+dist_max[1]/2)])
    x_comm=torch.zeros((n,2))
    theta = torch.linspace(0, np.pi*2, 360)
    for i in range(n):
        phi=theta[np.random.randint(0, 360-1, 1)][0]
        x_comm[i,0] = dist_max[0]/2 * torch.cos(phi)+centro[0]
        x_comm[i,1] = dist_max[1]/2 * torch.sin(phi)+centro[1]

    return x_comm

def large_config_stamp_generator(comm_radio, task_agents, comm_agents):
    task_config = x_task_gen(r=comm_radio, n=task_agents)
    if task_agents>1:
       comm_config = x_comm_gen(r=comm_radio,n=comm_agents, task_agents=task_config)
    else:
       comm_config = x_task_gen(r=comm_radio, n=comm_agents) 
    
    # Definir el ángulo de rotación en radianes (por ejemplo, 45 grados)
    angulo=torch.rand(1)*2*torch.pi
    # Matriz de rotación en sentido antihorario en 2D
    matriz_rotacion = torch.tensor([[torch.cos(angulo), -torch.sin(angulo)],
                                [torch.sin(angulo), torch.cos(angulo)]],dtype=torch.double)

    task_config=torch.matmul(task_config-torch.tensor([comm_radio,comm_radio])/2,matriz_rotacion)+torch.tensor([comm_radio,comm_radio])/2
    comm_config=torch.matmul(comm_config-torch.tensor([comm_radio,comm_radio])/2,matriz_rotacion)+torch.tensor([comm_radio,comm_radio])/2
    
    return task_config, comm_config

def gen_large_config_from_stamps(comm_radio, task_agents, comm_agents):
    if comm_agents>1:
        num_stamps=min(task_agents//3,comm_agents//2)
    else:
        num_stamps=1
    stamp_task_agents=task_agents//num_stamps
    stamp_comm_agents=comm_agents//num_stamps
    rest_comm_agents=comm_agents%num_stamps
    rest_task_agents=task_agents%num_stamps
    resto=rest_comm_agents+rest_task_agents
       
    task_agents_config=torch.zeros((task_agents,2))
    comm_agents_config=torch.zeros((comm_agents,2))
    stamp_grid=math.ceil(math.sqrt(num_stamps+1))
    delta=[]
    for i in range(stamp_grid):
        for j in range(stamp_grid):
            delta.append([i,j])
    #fig, ax = plt.subplots(stamp_grid, stamp_grid, figsize=(5, 5))

        
    for stamp in range(num_stamps):
        shift=torch.tensor(delta[stamp])*comm_radio
        shift_vector_task=shift.repeat(stamp_task_agents, 1).reshape(stamp_task_agents, -1)
        shift_vector_comm=shift.repeat(stamp_comm_agents, 1).reshape(stamp_comm_agents, -1)
        stamp_task_config, stamp_comm_config = large_config_stamp_generator(comm_radio, stamp_task_agents, stamp_comm_agents)
       
        task_agents_config[stamp*stamp_task_agents:(stamp+1)*stamp_task_agents]=stamp_task_config+shift_vector_task
        comm_agents_config[stamp*stamp_comm_agents:(stamp+1)*stamp_comm_agents]=stamp_comm_config+shift_vector_comm
        #plot_config(config=torch.vstack((task_agents_config[stamp*stamp_task_agents:(stamp+1)*stamp_task_agents], comm_agents_config[stamp*stamp_comm_agents:(stamp+1)*stamp_comm_agents])),ax=ax[delta[stamp][0],delta[stamp][1]],task_ids=np.arange(stamp_task_agents))
        
    if resto>0:
        stamp_task_config, stamp_comm_config = large_config_stamp_generator(comm_radio, rest_task_agents, rest_comm_agents)
        shift=torch.tensor(delta[num_stamps])*comm_radio

        if rest_task_agents>0:
            shift_vector_task=shift.repeat(rest_task_agents, 1).reshape(rest_task_agents, -1)
            task_agents_config[(num_stamps)*stamp_task_agents:]=stamp_task_config+shift_vector_task

        if rest_comm_agents>0:
            shift_vector_comm=shift.repeat(rest_comm_agents, 1).reshape(rest_comm_agents, -1)
            comm_agents_config[(num_stamps)*stamp_comm_agents:]=stamp_comm_config+shift_vector_comm
    
        #plot_config(config=torch.vstack((task_agents_config[(num_stamps)*stamp_comm_agents:], comm_agents_config[(num_stamps)*stamp_comm_agents:])),ax=ax[delta[num_stamps][0],delta[num_stamps][1]],task_ids=np.arange(rest_task_agents))

    #plt.show()
    return task_agents_config, comm_agents_config

def generate_large_samples(params, t0):
    df_train=[]
    df_val=[]
    sample_count = params['sample_count']
    comm_radio=(params['channel'].rango)*1.1
    for mode in ('train', 'test'):
        for i in range(sample_count[mode]):
            status = 'infeasible'
            
            while (status != 'optimal'):
                
                params['task_agents']=torch.randint(low=math.ceil(params['total_agents'] /2), high=math.ceil(2*params['total_agents']/3), size=(1,)).item()
                params['comm_agents']=params['total_agents']-params['task_agents']
                task_config, comm_config = gen_large_config_from_stamps(comm_radio, params['task_agents'], params['comm_agents'])
                config=np.vstack((task_config.numpy(), comm_config.numpy()))
                adj=params['channel'].adjacency(config)
                rate,_=params['channel'].predict(config)
                try:
                    MFR, status=MNF_graph_solver(params['task_agents'], params['comm_agents'],adj,rate)
                except:	
                    status = 'infeasible' 

                if (status == 'optimal'):
                    if np.isnan(MFR) or np.isinf(MFR):
                        print(MFR)
                        status = 'infeasible'
                    else:
                        #'positions', 'types', 'edge_index','MFR'
                        data = points_to_data(task_config,comm_config, MFR)
                        if mode=='train':
                            df_train.append(data)
                        if mode=='test':
                            df_val.append(data)
                        
                        duration_str = human_readable_duration(time.time()-t0)
                    msg =console_width_str(f'generated {i} {mode} samples in {duration_str}')
                    print('\r' + msg + '\r', end='')                
            
    with open(params['filename']+'_train.pt', 'wb') as f:
        pk.dump(df_train, f)
    
    with open(params['filename']+'_val.pt', 'wb') as f:
        pk.dump(df_val, f)

def generate_samples(params, t0):
    df_train=[]
    df_val=[]
    sample_count = params['sample_count']
    Kopts=np.arange(params['task_agents']*(params['task_agents']-1))
    Kopts=np.arange(params['task_agents']*(params['task_agents']-1))
    mfr=MNF_share_solver(num_task_config=params['task_agents'], num_comm_config=params['comm_agents'], channel=params['channel'], Kopts=Kopts)
    comm_radio=(params['channel'].rango)*1.0
    for mode in ('train', 'test'):
        for i in range(sample_count[mode]):
            status = 'infeasible'
            
            while (status != 'optimal'):

                task_config = x_task_gen(comm_radio, params['task_agents'])
                comm_config = x_comm_gen(comm_radio,params['comm_agents'], task_config)
                #config=np.vstack((task_config.numpy(), comm_config.numpy()))
                #adj=params['channel'].adjacency(config)
                #rate,_=params['channel'].predict(config)
                try:
                    MFR, status,ai,Tau, rs=mfr.solve(task_config=task_config.numpy(), comm_config=comm_config.numpy())
                except:	
                    status = 'infeasible' 

                if (status == 'optimal'):
                    if np.isnan(MFR) or np.isinf(MFR):
                        print(MFR)
                        status = 'infeasible'
                    else:
                        #'positions', 'types', 'edge_index','MFR'
                        data = points_to_data(task_config,comm_config, MFR)

                        if mode=='train':
                            df_train.append(data)
                        if mode=='test':
                            df_val.append(data)
                        
                        duration_str = human_readable_duration(time.time()-t0)
                    msg =console_width_str(f'generated {i} {mode} samples in {duration_str}')
                    print('\r' + msg + '\r', end='')

    with open(params['filename']+'_train.pt', 'wb') as f:
        pk.dump(df_train, f)
    
    with open(params['filename']+'_val.pt', 'wb') as f:
        pk.dump(df_val, f)

def create_large_data_set(args):
    samples = args.samples
    outDir = args.outDir
    tt = time.time()
    params={}
    train_samples = int(0.85 * samples)
    params['sample_count'] = {'train': train_samples, 'test': samples - train_samples}
    params['total_agents'] = args.total_count
    params['output_dir'] = outDir
    params['var_array'] = args.var_array
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    params['channel'] = expModel(indicatrix=True)
    params['filename'] = params['output_dir'] +  f"graph_ind_{params['total_agents']}_{timestamp}{params['var_array']}"
    generate_large_samples(params,tt)
    duration_str = human_readable_duration(time.time()-tt)
    print(f'generated {samples} samples in {duration_str}')
    print(f'saved data to: {params["filename"]}')

def create_data_set(args):
    samples = args.samples
    outDir = args.outDir
    tt = time.time()
    params={}
    train_samples = int(0.85 * samples)
    params['sample_count'] = {'train': train_samples, 'test': samples - train_samples}
    params['task_agents'] = args.task_count
    params['comm_agents'] = args.comm_count
    params['output_dir'] = outDir
    params['var_array'] = args.var_array
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if args.ind:
        params['channel'] = expModel(indicatrix=True)
        params['filename'] = params['output_dir'] +  f"graph_ind_{params['task_agents']}t_{params['comm_agents']}c_{timestamp}{params['var_array']}"
    else:
        params['alpha'] = args.alpha
        params['channel'] = expModel(alpha=params['alpha'])
        params['filename'] = params['output_dir'] +  f"graph_{int(params['alpha']*10)}_{params['task_agents']}t_{params['comm_agents']}c_{timestamp}{params['var_array']}"
    generate_samples(params,tt)
    duration_str = human_readable_duration(time.time()-tt)
    print(f'generated {samples} samples in {duration_str}')
    print('saved data to: ' + params['filename'])


def view_dataset(args):
    data_path = args.file
    samples_in = args.samples
    dataset = Path(data_path)
    #filename = data_path.split('/')[-1]
    #alpha =float(filename.split('_')[1])/10
    #channel=expModel(alpha=alpha)
    arena_side=20#(channel.rango/2.0)*1.8
    if not dataset.exists():
        print(f'the dataset {dataset} was not found')
        return
    
    with open(data_path, 'rb') as f:
        view_set = pk.load(f)

    if isinstance(samples_in, list):
        samples=samples_in
    else:    
        samples = np.random.randint(low=0, high=len(view_dataset), size=samples_in)
    
    for i in samples:
        data=view_set[i]
        graph_plot(data)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='utilities for hdf5 datasets for learning connectivity')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # generate subparser
    gen_parser = subparsers.add_parser('generate', help='generate dataset')
    gen_parser.add_argument('alpha', type=float, help='adj parameter')
    gen_parser.add_argument('samples', type=int, help='number of samples to generate')
    gen_parser.add_argument('task_count', type=int, help='number of task agents')
    gen_parser.add_argument('comm_count', type=int, help='number of comm agents')
    gen_parser.add_argument('--var_array', type=str, default = '',help='number of array')
    gen_parser.add_argument('--outDir', '-oD', type=str, default=str(Path(__file__).resolve().parent)+'/',
                            help='path to output dir location, default parent folder')
    gen_parser.add_argument('--ind', type=bool, default = False,help='indicatrix')


    # large generate subparser
    gen_parser = subparsers.add_parser('large_generate', help='generate large config dataset')
    gen_parser.add_argument('samples', type=int, help='number of samples to generate')
    gen_parser.add_argument('total_count', type=int, help='number of total agents')
    gen_parser.add_argument('--var_array', type=str, default = '',help='number of array')
    gen_parser.add_argument('--outDir', '-oD', type=str, default=str(Path(__file__).resolve().parent)+'/',
                            help='path to output dir location, default parent folder')
    
      
    view_parser = subparsers.add_parser('view', help='view data from dataset')
    view_parser.add_argument('samples', type=int, help='number of samples to generate',default=5)
    view_parser.add_argument('--file', type=str, help='path to dataset file')

    args = parser.parse_args()

    if args.command == 'generate':
        create_data_set(args)
    elif args.command == 'large_generate':
        create_large_data_set(args)

    elif args.command == 'view':
        view_dataset(args)