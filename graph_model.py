import numpy as np
import torch
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl
import torch.nn.functional as F
from egnn_pytorch import EGNN_Sparse
from graph_dataset_utils import  points_to_data
from utils.channel_model import expModel   

class GCN_EG(torch.nn.Module):
    def __init__(self, output_dims,m_dims,update_coors,NormCoors,aggr, dropout):
        torch.manual_seed(12345)
        super().__init__()
        self.conv1 = EGNN_Sparse(feats_dim=2,pos_dim=2,update_coors=update_coors,m_dim=m_dims[0],out_dim=output_dims[0], dropout=dropout, aggr=aggr,norm_coors=NormCoors)
        self.conv2 = EGNN_Sparse(feats_dim=output_dims[0],pos_dim=2,update_coors=update_coors,m_dim=m_dims[1],out_dim=output_dims[1], dropout=dropout, aggr=aggr,norm_coors=NormCoors)
        self.conv3 = EGNN_Sparse(feats_dim=output_dims[1],pos_dim=2,update_coors=update_coors,m_dim=m_dims[2],out_dim=output_dims[2], dropout=dropout, aggr=aggr,norm_coors=NormCoors)
        #self.conv4 = TAGConv(2, hidden_channels,K=K)
        self.lin = Linear(output_dims[2],out_features=1)
        self.dropout=dropout

        #self.aggr = MeanAggregation()
    def forward(self, x, edge_index, edge_attr,positions, batch):
        # 1. Obtain node embeddings 
        xt=torch.cat([positions, x], dim=-1).float()
        #xt=torch.hstack((positions, positions)).float()
        x = self.conv1(xt, edge_index, edge_attr,batch)
        #x = x.relu()
        x = self.conv2(x, edge_index, edge_attr,batch)
        #x = x.relu()
        x = self.conv3(x, edge_index, edge_attr,batch)
  
        x_=x[:,2:]
       
        #x_=self.conv4(x_, edge_index)
        # 2. Readout layer
       
        x_ = global_mean_pool(x_, batch) 
        # [batch_size, hidden_channels]
        
        # 3. Apply a final classifier
        #x = F.dropout(x, p=self.dropout, training=self.training)
        x_ = self.lin(x_)
        
        return x_
    


class LightningEGNN_net(pl.LightningModule):
    def __init__(self,optimizer,learning_rate, output_dims,m_dims,update_coors, NormCoors, aggr, dropout, **kwargs):
        super().__init__()
        self.model = GCN_EG(output_dims=output_dims,m_dims=m_dims,update_coors=update_coors, NormCoors=NormCoors, aggr=aggr, dropout=dropout)
        self.loss_module = torch.nn.SmoothL1Loss()
        self.map_metric = torchmetrics.MeanAbsolutePercentageError()
        self.R2metric = torchmetrics.R2Score(num_outputs=1)
        self.selfoptimizer=optimizer
        self.lr=learning_rate
        self.save_hyperparameters()
        
    def forward(self, x, edge_index,edge_attr, positons, batch):
        return self.model(x, edge_index,edge_attr, positons, batch)
    
    def training_step(self, batch, batch_idx):
        x, edge_index, edge_attr, positions, batch_= batch.x, batch.edge_index,batch.edge_attr, batch.pos, batch.batch 
        y_pred= self.forward(x, edge_index, edge_attr,positions, batch_)
        y_pred=y_pred.reshape(y_pred.shape[0])
        loss = self.loss_module(y_pred, batch.y)
        #r2_m = self.R2metric(y_pred, batch.y)
        map_m=self.map_metric(y_pred, batch.y)
        
        self.log("train_loss", loss, batch_size=y_pred.shape[0])
        #self.log("train_R2", r2_m)
        self.log("train_MAPE", map_m,batch_size=y_pred.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_index, edge_attr, positions, batch_= batch.x, batch.edge_index,batch.edge_attr, batch.pos, batch.batch       
        with torch.no_grad():
            y_pred= self.forward(x, edge_index, edge_attr,positions, batch_)
            y_pred=y_pred.reshape(y_pred.shape[0],)            
            loss = self.loss_module(y_pred, batch.y)
            #r2_m = self.R2metric(y_pred, batch.y)
            map_m=self.map_metric(y_pred, batch.y)
        
        self.log("val_loss", loss,batch_size=y_pred.shape[0])
        #self.log("val_R2", r2_m)
        self.log("val_MAPE", map_m,batch_size=y_pred.shape[0])
    
    def test_step(self, batch, batch_idx):
        x, edge_index, edge_attr, positions, batch_= batch.x, batch.edge_index,batch.edge_attr, batch.pos, batch.batch       
        with torch.no_grad():
            y_pred= self.forward(x, edge_index, edge_attr,positions, batch_)
            y_pred=y_pred.reshape(y_pred.shape[0],)            
            loss = self.loss_module(y_pred, batch.y)
            #r2_m = self.R2metric(y_pred, batch.y)
            map_m=self.map_metric(y_pred, batch.y)
        
        self.log("test_loss", loss,batch_size=y_pred.shape[0])
        #self.log("val_R2", r2_m)
        self.log("test_MAPE", map_m,batch_size=y_pred.shape[0])
        
    def evaluate(self, data):
        x, edge_index, edge_attr, positions, batch_= data.x, data.edge_index,data.edge_attr, data.pos, data.batch  
        with torch.no_grad():
            y_pred= self.forward(x, edge_index, edge_attr,positions, batch_)
        
        return y_pred
    
    def configure_optimizers(self):
        if self.selfoptimizer == "sgd":
            optimizer = optim.SGD(self.model.parameters(),
                              lr=self.lr, momentum=0.9)
        elif self.selfoptimizer == "adam":
            optimizer = optim.Adam(self.model.parameters(),
                               lr=self.lr)
        return optimizer
    

def grad_simulate_step(NA, model, TA, device, lr=0.3) :
    NA=torch.from_numpy(NA)
    NA.requires_grad=True
    data=points_to_data(TA,NA).to(device)
    xt, edge_index, edge_attr, positions, batch= data.x, data.edge_index, data.edge_attr, data.pos, data.batch
    f_x=model.forward(xt, edge_index, edge_attr,positions,batch)
    f_x.backward()
    #d=(NA.grad/torch.linalg.norm(NA.grad))
    grad_f_x=NA.grad
    return grad_f_x*lr, f_x.cpu().detach().numpy()


def optimize_NA(NA, model, TA, device, max_iter=500, lr=0.3):
    dxy=np.ones_like(NA)
    iter=0
    NA_history=[]
    NA_history.append(NA)
    # Simulation loop
    while (iter < max_iter) and (np.sum(np.linalg.norm(dxy))>1e-3):
        iter+=1
        dxy=grad_simulate_step(NA_history[-1], model=model, TA=TA, device=device,lr=lr)
        NA_history.append(NA_history[-1]+dxy.numpy())

    if iter==max_iter:
        print("salio por iteraciones")
    else:
        print(f"salio por norma en {iter} iteraciones")
    return NA_history[-1]

def evaluar_grilla_model(task_config,model,device):
    canal=expModel(indicatrix=True)
    dist=(canal.rango)*1.0
    rango=0.5
    x=np.linspace(rango,(dist-rango),int(2*((dist))+1))
    y=np.linspace(rango,(dist-rango),int(2*((dist))+1))
    NA=np.array([[0.5,0.5]])
    c_map=np.empty((len(x),len(y)))
    for c_i, i in enumerate(x):
        for c_j, j in enumerate(y):
            NA[0,0]=i
            NA[0,1]=j
            #print(NA)
            try: 
                data=points_to_data(task_config,torch.from_numpy(NA)).to(device)
                c_map[c_i,c_j]=model.evaluate(data).cpu().detach().numpy()
            except:
                c_map[c_i,c_j]=-4

    return np.max(c_map)