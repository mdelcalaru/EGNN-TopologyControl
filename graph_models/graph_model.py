# graph_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau

from egnn_pytorch import EGNN_Sparse
from graph_models.losses import MyGradLoss2, my_grad_loss
from graph_models.metrics import MyMAPE_Metric, MyMAE_Metric
import torchmetrics

class EGNN_base_network(nn.Module):
    def __init__(self, output_dims, m_dims, update_coors, update_feats, NormCoors, aggr, dropout, edges):
        super().__init__()
        torch.manual_seed(12345)

        edge_attr_dim = 1 if edges else 0
        self.layers = nn.ModuleList()
        in_feats_dim = 2  # one-hot encoded features

        for i in range(len(m_dims)):
            self.layers.append(EGNN_Sparse(
                feats_dim=in_feats_dim,
                pos_dim=2,
                edge_attr_dim=edge_attr_dim,
                update_coors=update_coors,
                update_feats=update_feats,
                m_dim=m_dims[i],
                out_dim=output_dims[i],
                dropout=dropout,
                aggr=aggr,
                norm_coors=NormCoors,
                coor_weights_clamp_value=True
            ))
            if update_feats:
                in_feats_dim = output_dims[i]

class EGNN_network(pl.LightningModule):
    def __init__(self, mode='grad', optimizer='adam', learning_rate=1e-3,
                 output_dims=None, m_dims=None, update_coors=True, update_feats=True,
                 NormCoors=False, aggr='add', dropout=0.0, edges=False,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.mode = mode  # 'grad' or 'mnf'
        # print('-------------------------------------------------------------------------')
        # print(f"Initializing EGNN_network in {self.mode} mode with edges={edges}")
        # print(f"Optimizer: {optimizer}, Learning Rate: {learning_rate}")
        # print(f"Output Dims: {output_dims}, M Dims: {m_dims}, Update Coors: {update_coors}, Update Feats: {update_feats}")
        # print(f"Norm Coors: {NormCoors}, Aggregation: {aggr}, Dropout: {dropout}")
        # print('-------------------------------------------------------------------------')
        # Initialize the EGNN base network
        if output_dims is None or m_dims is None:
            raise ValueError("output_dims and m_dims must be provided")
        self.model = EGNN_base_network(output_dims, m_dims, update_coors, update_feats,
                                       NormCoors, aggr, dropout, edges)

        if self.mode == 'mnf':
            self.lin = Linear(output_dims[-1] if update_feats else 2, out_features=1)
            self.loss_module = nn.SmoothL1Loss(reduction='mean')
            self.map_metric = torchmetrics.MeanAbsolutePercentageError()
        else:
            self.grad_loss_module = MyGradLoss2()
            self.mape = MyMAPE_Metric()
            self.mae = MyMAE_Metric()

        self.optimizer = optimizer
        self.lr = learning_rate
        self.edges = edges

    def forward(self, x, edge_index, edge_attr, batch, positions):
        xt = torch.cat([positions, x], dim=-1).float()
        edge_attrNet = edge_attr.float() if self.edges else None

        for layer in self.model.layers:
            xt = layer(xt, edge_index, edge_attrNet, batch)

        if self.mode == 'grad':
            return xt[:, :2]
        else:
            mnf = xt[:, 2:]
            mnf = global_mean_pool(mnf, batch)
            return self.lin(mnf)

    def training_step(self, batch, batch_idx):
        if self.mode == 'grad':
            g_x = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.pos)
            loss = self.grad_loss_module(g_pred=g_x, grafo=batch)
            self.log("train_loss", loss, batch_size=g_x.shape[0])
            return loss
        else:
            y_pred = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.pos)
            y_pred = y_pred.reshape(-1)
            loss = self.loss_module(y_pred, batch.y)
            self.log("train_loss", loss, batch_size=y_pred.shape[0])
            self.log("train_MAPE", self.map_metric(y_pred, batch.y), batch_size=y_pred.shape[0])
            return loss

    def validation_step(self, batch, batch_idx):
        if self.mode == 'grad':
            g_x = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.pos)
            loss = self.grad_loss_module(g_pred=g_x, grafo=batch)
            self.mape.update(g_x, batch)
            self.mae.update(g_x, batch)
            self.log("val_loss", loss, batch_size=g_x.shape[0])
        else:
            y_pred = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.pos)
            y_pred = y_pred.view(-1)
            loss = self.loss_module(y_pred, batch.y)
            self.log("val_loss", loss, batch_size=y_pred.shape[0])
            self.log("val_MAPE", self.map_metric(y_pred, batch.y), batch_size=y_pred.shape[0])

    def on_validation_epoch_end(self):
        if self.mode == 'grad':
            self.log("val_mape", self.mape.compute())
            self.log("val_mae", self.mae.compute())
            self.mape.reset()
            self.mae.reset()

    def test_step(self, batch, batch_idx):
        if self.mode == 'grad':
            g_x = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.pos)
            loss = self.grad_loss_module(g_pred=g_x, grafo=batch)
            self.log("test_loss", loss, batch_size=g_x.shape[0])
        else:
            y_pred = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.pos)
            y_pred = y_pred.view(-1)
            loss = self.loss_module(y_pred, batch.y)
            self.log("test_loss", loss, batch_size=y_pred.shape[0])
            self.log("test_MAPE", self.map_metric(y_pred, batch.y), batch_size=y_pred.shape[0])

    def evaluate(self, data):
        xt, edge_index, edge_attr, positions, batch= data.x, data.edge_index, data.edge_attr, data.pos, data.batch
        with torch.no_grad():
            return self.forward(xt, edge_index, edge_attr, batch, positions)

    def configure_optimizers(self):
        if self.optimizer == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10)

        return {"optimizer": optimizer,
               "lr_scheduler": {
                 "scheduler": scheduler,
                 "monitor": "val_loss",
                 "interval": "epoch",
                 "frequency": 1
             },
        }
        #return optimizer
        
def load_models(model_y_file, model_file_grad, device):
    model_y = EGNN_network.load_from_checkpoint(model_y_file).to(device)
    model_grad= EGNN_network.load_from_checkpoint(model_file_grad).to(device)
    return model_y, model_grad

def load_legacy_checkpoint(checkpoint_path, mode: str, config, verbose=True):
    """
    Carga un checkpoint entrenado con las clases antiguas (gradient_EGNN_network o MNF_EGNN_network)
    dentro del nuevo modelo unificado EGNN_network.

    Params:
        checkpoint_path: ruta al .ckpt
        mode: "grad" o "mnf"
        config: diccionario de configuración
        verbose: si True, imprime detalles sobre parámetros no cargados

    Returns:
        modelo instanciado (EGNN_network) con pesos cargados
    """
    #from graph_model import EGNN_network  # por si lo tenés en otro archivo

    ckpt = torch.load(checkpoint_path)
    state_dict = ckpt["state_dict"]
    if "mode" in config.keys():
        model = EGNN_network(config=config)
    else:
        model = EGNN_network(config=config, mode=mode)
    model_state_dict = model.state_dict()

    # Intentamos cargar con matching parcial
    missing_keys = []
    unexpected_keys = []

    # Filtramos solo los keys que están en el modelo actual
    compatible_dict = {
        k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape
    }

    # Cargamos los parámetros compatibles
    model.load_state_dict(compatible_dict, strict=False)

    # Log
    for k in model_state_dict:
        if k not in state_dict:
            missing_keys.append(k)
    for k in state_dict:
        if k not in model_state_dict:
            unexpected_keys.append(k)

    if verbose:
        print("🔁 Carga del checkpoint anterior completada")
        print(f"✅ Parámetros cargados: {len(compatible_dict)}")
        print(f"❗ Faltaron: {len(missing_keys)} → {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        print(f"⚠️ Sobraron: {len(unexpected_keys)} → {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")

    return model


class gradient_EGNN_network(pl.LightningModule):
    def __init__(self,optimizer,learning_rate, output_dims,m_dims,update_coors,update_feats,
                 NormCoors=False,aggr='add', dropout=0.0, edges=False, **kwargs):
        super().__init__()
        
        self.model = EGNN_base_network(output_dims=output_dims,
                                       m_dims=m_dims,
                                       update_coors=True,
                                       update_feats=update_feats, 
                                       NormCoors=NormCoors, 
                                       aggr=aggr, 
                                       dropout=dropout,
                                       edges=edges)
        
        if update_feats:
            self.lin = Linear(output_dims[-1],out_features=1)
        else:
            self.lin = Linear(2,out_features=1)
            
        self.edges=edges
        self.grad_loss_module = my_grad_loss()
        self.mnf_loss_module = torch.nn.SmoothL1Loss(reduction='mean')
        self.mape = MyMAPE_Metric()
        self.mae = MyMAE_Metric()
        self.optimizer=optimizer
        self.lr=learning_rate
        self.save_hyperparameters()
        
        
    def forward(self,  x, edge_index, edge_attr, batch, positions):
        xt=torch.cat([positions, x], dim=-1).float() #xt is the input for the first layer it has the nodes positions and features
        
        if self.edges:
            edge_attrNet=edge_attr.float()
        else:
            edge_attrNet=None
            
        for layer in self.model.layers:
            xt = layer(xt, edge_index, edge_attrNet, batch)
        
        gradient=xt[:,:2]     
        mnf=xt[:,2:]
        mnf= global_mean_pool(mnf, batch) 
        #mnf= global_add_pool(mnf, batch)
        mnf = self.lin(mnf) 
           
        return gradient, mnf        
            
    def training_step(self, batch, batch_idx):        
        g_x ,mnf      = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.pos)
        loss_grad = self.grad_loss_module(g_pred=g_x, grafo=batch)
        #loss_mnf = self.mnf_loss_module(mnf.reshape(mnf.shape[0]), batch.y)
        self.log("train_loss", loss_grad, batch_size=g_x.shape[0])

        return loss_grad
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            g_x,mnf   = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.pos)
            loss_grad = self.grad_loss_module(g_pred=g_x, grafo=batch)
            self.mape.update(g_x, batch)
            self.mae.update(g_x, batch)
        #self.log("val_rel_grad_error", self.rel_grad_error, prog_bar=True)
        
        self.log("val_loss", loss_grad, batch_size=g_x.shape[0])
        
    def on_validation_epoch_end(self):
        mape_value = self.mape.compute()
        mae_value = self.mae.compute()
        self.log("val_mape", mape_value)
        self.log("val_mae", mae_value)
        self.mape.reset()
        self.mae.reset()    
        
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            g_x       = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.pos)
            loss_grad = self.grad_loss_module(g_pred=g_x, grafo=batch)
            mape_grad = self.mape_loss_module(g_x, batch)

        self.log("test_loss", loss_grad,batch_size=g_x.shape[0])
        self.log("test_MAE", mape_grad,batch_size=g_x.shape[0])
            
    def evaluate(self, batch):
        with torch.no_grad():
            g_x, mnf= self.forward( batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.pos)
        
        return g_x
    
    def configure_optimizers(self):
        if self.optimizer == "sgd":
            optimizer = optim.SGD(self.model.parameters(),
                              lr=self.lr, momentum=0.9)
        elif self.optimizer == "adam":
            optimizer = optim.Adam(self.model.parameters(),
                               lr=self.lr)
        
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "val_loss",  # Reduce LR basado en la pérdida de validación
        #         "interval": "epoch",
        #         "frequency": 1
        #     }  , 
        # } 
        return optimizer

'''Model that fits MNF optimization problem's solution''' 

class MNF_EGNN_network(pl.LightningModule):
    def __init__(self,optimizer,learning_rate, output_dims,m_dims,update_coors,update_feats,
                 NormCoors=False,aggr='add', dropout=0.0, edges=False, **kwargs):
        super().__init__()
        self.model = EGNN_base_network(output_dims=output_dims,
                                       m_dims=m_dims,
                                       update_coors=update_coors,
                                       update_feats=update_feats, 
                                       NormCoors=NormCoors, 
                                       aggr=aggr, 
                                       dropout=dropout,
                                       edges=edges)
        
        if update_feats:
            self.lin = Linear(output_dims[-1],out_features=1)
        else:
            self.lin = Linear(2,out_features=1)
            
        self.loss_module = torch.nn.SmoothL1Loss(reduction='mean')
        #self.loss_module = torch.nn.MSELoss(reduction='mean')
        self.map_metric = torchmetrics.MeanAbsolutePercentageError()
        self.optimizer=optimizer
        self.lr=learning_rate
        self.edges=edges
        self.save_hyperparameters()
        
    def forward(self, x, edge_index, edge_attr,positions, batch):
        
        xt=torch.cat([positions, x], dim=-1).float() #xt is the input for the first layer it has the nodes positions and features
        if self.edges:
            edge_attrNet=edge_attr.float()
        else:
            edge_attrNet=None
            
        for layer in self.model.layers:
            xt = layer(xt, edge_index, edge_attrNet, batch)
        
        mnf=xt[:,2:]
        mnf= global_mean_pool(mnf, batch) 
        #mnf= global_add_pool(mnf, batch)
        mnf = self.lin(mnf) 
        
        return mnf        
        
    
    def training_step(self, batch, batch_idx):
        x, edge_index, edge_attr, positions, batch_= batch.x, batch.edge_index,batch.edge_attr, batch.pos, batch.batch 
        y_pred= self.forward(x, edge_index, edge_attr,positions, batch_)
        y_pred=y_pred.reshape(y_pred.shape[0])
        loss = self.loss_module(y_pred, batch.y)
        map_m=self.map_metric(y_pred, batch.y)
        
        self.log("train_loss", loss, batch_size=y_pred.shape[0])
        self.log("train_MAPE", map_m,batch_size=y_pred.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_index, edge_attr, positions, batch_= batch.x, batch.edge_index,batch.edge_attr, batch.pos, batch.batch       
        with torch.no_grad():
            y_pred= self.forward(x, edge_index, edge_attr,positions, batch_)
            y_pred=y_pred.reshape(y_pred.shape[0],)            
            loss = self.loss_module(y_pred, batch.y)
            map_m=self.map_metric(y_pred, batch.y)
        
        self.log("val_loss", loss,batch_size=y_pred.shape[0])
        self.log("val_MAPE", map_m,batch_size=y_pred.shape[0])
        
        
    
    def test_step(self, batch, batch_idx):
        x, edge_index, edge_attr, positions, batch_= batch.x, batch.edge_index,batch.edge_attr, batch.pos, batch.batch       
        with torch.no_grad():
            y_pred= self.forward(x, edge_index, edge_attr,positions, batch_)
            y_pred=y_pred.reshape(y_pred.shape[0],)            
            loss = self.loss_module(y_pred, batch.y)
            map_m=self.map_metric(y_pred, batch.y)
        
        self.log("test_loss", loss,batch_size=y_pred.shape[0])
        self.log("test_MAPE", map_m,batch_size=y_pred.shape[0])
        
    def evaluate(self, data):
        x, edge_index, edge_attr, positions, batch_= data.x, data.edge_index,data.edge_attr, data.pos, data.batch  
        with torch.no_grad():
            y_pred= self.forward(x, edge_index, edge_attr,positions, batch_)
        
        return y_pred
    
    def configure_optimizers(self):
        if self.optimizer == "sgd":
            optimizer = optim.SGD(self.model.parameters(),
                              lr=self.lr, momentum=0.9)
        elif self.optimizer == "adam":
            optimizer = optim.Adam(self.model.parameters(),
                               lr=self.lr)#,weight_decay=1e-4)
            
            
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Reduce LR basado en la pérdida de validación
                "interval": "epoch",
                "frequency": 1
            },
        }

def load_models2(model_y_file, model_file_grad, device):
    model_y = EGNN_network.load_from_checkpoint(model_y_file).to(device)
    model_grad= gradient_EGNN_network.load_from_checkpoint(model_file_grad).to(device)
    return model_y, model_grad