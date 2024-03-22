import argparse
from pathlib import Path
import datetime
import time
import h5py
import torch
from mid.utils import human_readable_duration, console_width_str, graph_plot
from mid.MFR_cvxpy import MFR_share_solver
import numpy as np
from mid.channel_model import expModel
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import glob
import pickle as pk
from torch_geometric.loader import DataLoader

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

def write_data_file(filename, data_dict):
    df_train=[]
    df_val=[]

    for idx in range(len(data_dict)):
        # for each sample
        mode=data_dict[idx]['mode']     
        data=data_dict[idx]['data']

        if mode=='train':
            df_train.append(data)
        else:
            df_val.append(data)

    with open(filename+'_train.pt', 'wb') as f:
        pk.dump(df_train, f)
    
    with open(filename+'_val.pt', 'wb') as f:
        pk.dump(df_val, f)

def points_to_data(TA,NA,mfr=None):
    task_agents=TA.shape[0]
    comm_agents=NA.shape[0]
    positions = torch.vstack((TA,NA))
    types=torch.zeros((task_agents+comm_agents,2))
    types[0:task_agents,0]=1
    types[task_agents:,1]=1
    edge_index = edge_index_gen(task_agents,comm_agents)

    edge_index=to_undirected(edge_index)
    if mfr is None:
        data=Data(x=types, edge_index=edge_index, pos=positions)
    else:
        data=Data(x=types, edge_index=edge_index, pos=positions, y=mfr)
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
            x_task[i,0] =  r * rng.uniform(low=0.1,high=1.0,size=None)
            x_task[i,1] =  r * rng.uniform(low=0.1,high=1.0,size=None)#*np.random.choice([-1,1])
        
        x_task[1,0] = r * rng.uniform(low=0.8,high=1.2,size=None) 

    return x_task

def x_comm_gen(r,n, task_agents, rndm_seed=None):
    rng = np.random.default_rng(seed=rndm_seed)
    max_task_agents= torch.max(task_agents, dim=0).values.tolist()
    min_task_agents= torch.min(task_agents, dim=0).values.tolist()
    m=(max_task_agents[1]-min_task_agents[1])/(max_task_agents[0]-min_task_agents[0])
    b=min_task_agents[1]-m*min_task_agents[0]
    dist_max=np.max(np.array([max_task_agents[0]-min_task_agents[0],max_task_agents[1]-min_task_agents[1]]))

    x_comm=torch.zeros((n,2))
    for i in range(n):
        x_comm[i,0] =  rng.uniform(low=min_task_agents[0],high=max_task_agents[0],size=None)
        margen=rng.uniform(low=0.01,high=1.0,size=None)*dist_max/2
        x_comm[i,1] = m * (x_comm[i,0]+margen) + b 

    return x_comm

    
def generate_samples(params, t0):
    sample_count = params['sample_count']
    Kopts=np.arange(params['task_agents']*(params['task_agents']-1))
    data_dict = []

    for mode in ('train', 'test'):
        for i in range(sample_count[mode]):
            status = 'infeasible'
            
            while (status != 'optimal'):
                comm_radio=(params['channel'].rango)*1.5
                task_config = x_task_gen(comm_radio, params['task_agents'])
                #task_config = x_circle(comm_radio, params['task_agents'])
                comm_config = x_comm_gen(comm_radio,params['comm_agents'], task_config)
                
                try: 
                    mfr = MFR_share_solver(task_config.numpy(), comm_config.numpy(), params['channel'], Kopts)
                    MFR ,rs, ai, Tau, status=mfr.solver()
                except:	
                    status = 'infeasible' 

                if (status == 'optimal'):
                    if np.isnan(MFR) or np.isinf(MFR):
                        print(MFR)
                        status = 'infeasible'
                    else:
                        #'positions', 'types', 'edge_index','MFR'
                        data = points_to_data(task_config,comm_config, MFR)
                        data_dict.append({'mode': mode, 'data': data})
                        
                        duration_str = human_readable_duration(time.time()-t0)
                    msg =console_width_str(f'generated {len(data_dict)} samples in {duration_str}')
                    print('\r' + msg + '\r', end='')

    return data_dict



def edge_index_gen(task,comm):
    M=task+comm
    map1=torch.zeros(int(M*(M-1)/2),dtype=torch.int64)
    map2=torch.zeros(int(M*(M-1)/2),dtype=torch.int64)
    count=0
    for i in range(M):
        for j in range(i+1,M):
            map1[count]=int(i)
            map2[count]=int(j)
            count+=1
    #ed1=torch.hstack((map1,map2))
    #ed2=torch.hstack((map2,map1))
    return torch.vstack((map1,map2))


def create_data_set(args):
    alpha = args.alpha
    samples = args.samples
    task_agents = args.task_count
    comm_agents = args.comm_count
    outDir = args.outDir
    tt = time.time()
    params={}
    params['alpha'] = alpha
    train_samples = int(0.85 * samples)
    params['sample_count'] = {'train': train_samples, 'test': samples - train_samples}
    params['task_agents'] = task_agents
    params['comm_agents'] = comm_agents
    params['output_dir'] = outDir
    params['var_array'] = args.var_array
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if args.ind:
        params['channel'] = expModel(alpha=params['alpha'], indicatrix=True)
        filename = params['output_dir'] +  f"graph_ind_{params['task_agents']}t_{params['comm_agents']}c_{timestamp}{params['var_array']}"
    else:
        params['channel'] = expModel(alpha=params['alpha'])
        filename = params['output_dir'] +  f"graph_{int(params['alpha']*10)}_{params['task_agents']}t_{params['comm_agents']}c_{timestamp}{params['var_array']}"
    data_dict = generate_samples(params,tt)
    write_data_file(filename, data_dict)
    duration_str = human_readable_duration(time.time()-tt)
    print(f'generated {samples} samples in {duration_str}')
    print(f'saved data to: {filename}')

def add_to_data_set(args):
    data_path = args.file
    samples = args.samples
    dataset = Path(data_path)
    tt=time.time()
    if not dataset.exists():
        print(f'the dataset {dataset} was not found')
        return
    
    filename = data_path.split('/')[-1]
    outDir = data_path[:-len(filename)]
    params={}
    params['alpha'] =float(filename.split('_')[1])/10
    train_samples = int(0.85 * samples)
    params['sample_count'] = {'train': train_samples, 'test': samples - train_samples}
    params['task_agents'] = int(filename.split('_')[2][:-1])
    params['comm_agents'] = int(filename.split('_')[3][:-1])
    params['output_dir'] = outDir
    params['channel'] = expModel(alpha=params['alpha'])
    data_dict = generate_samples(params,tt)
    write_data_file(dataset, data_dict)
    duration_str = human_readable_duration(time.time()-tt)
    print(f'generetad {samples} samples in {duration_str}')
    print(f'added data to: {filename}')

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
        view_dataset = pk.load(f)

    if isinstance(samples_in, list):
        samples=samples_in
    else:    
        samples = np.random.randint(low=0, high=len(view_dataset), size=samples_in)
    
    for i in samples:
        data=view_dataset[i]
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
    
    add_to_set_parser = subparsers.add_parser('add', help='add data to dataset')
    add_to_set_parser.add_argument('samples', type=int, help='number of samples to generate')
    add_to_set_parser.add_argument('--file', type=str, help='path to output file')
    
    view_parser = subparsers.add_parser('view', help='view data from dataset')
    view_parser.add_argument('samples', type=int, help='number of samples to generate',default=5)
    view_parser.add_argument('--file', type=str, help='path to dataset file')

    args = parser.parse_args()

    if args.command == 'generate':
        create_data_set(args)
    elif args.command == 'add':
        add_to_data_set(args)
    elif args.command == 'view':
        view_dataset(args)