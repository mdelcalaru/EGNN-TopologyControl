import torch
import torch.nn as nn
import torch.nn.functional as F


class MyGradLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, g_pred, grafo):
        dif_pos = grafo.dif_pos       # [B*N, d, V]
        dif_y = grafo.dif_y           # [B, V]
        batch = grafo.batch           # [B*N]
        batch_size = dif_y.shape[0]

        total_loss = 0.0
        for i in range(batch_size):
            idx_i = (batch == i)
            g_i = g_pred[idx_i]             # [N, d]
            dpos_i = dif_pos[idx_i]         # [N, d, V]

            dot_per_node = torch.einsum('nd,ndv->nv', g_i, dpos_i)  # [N, V]
            y_pred = dot_per_node.sum(dim=0)                         # [V]
            y_true = dif_y[i]
            loss_i = torch.mean((y_pred - y_true) ** 2)
            total_loss += loss_i

        return total_loss / batch_size

class MyGradLoss2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, g_pred, grafo):
        # [B*N, d] @ [B*N, d, V] → [B*N, V]
        dot_per_node = torch.einsum('nd,ndv->nv', g_pred, grafo.dif_pos)

        # Queremos sumar sobre los nodos por grafo (usando batch)
        # Resultado: [B, V]
        y_pred = torch.zeros_like(grafo.dif_y)
        y_pred = y_pred.index_add(0, grafo.batch, dot_per_node)

        # Ahora computamos el MSE por grafo
        loss = F.mse_loss(y_pred, grafo.dif_y, reduction='mean')
        return loss

class my_grad_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, g_pred, grafo):
        dif_pos, dif_y = grafo.dif_pos, grafo.dif_y
        batch_size = grafo.batch_size
        batch = grafo.batch

        result = dif_pos * g_pred.unsqueeze(-1).repeat(1, 1, dif_pos.shape[-1])
        loss = 0
        for i in range(batch_size):
            sample_result = result[torch.where(batch == i)[0], :, :]
            loss += torch.mean((torch.sum(sample_result, dim=(0, 1)) - dif_y[i, :]) ** 2)

        return loss / batch_size


class my_mape_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, g_pred, grafo):
        eps = 1e-8
        dif_pos, dif_y = grafo.dif_pos, grafo.dif_y
        batch_size = grafo.batch_size
        batch = grafo.batch

        result = dif_pos * g_pred.unsqueeze(-1).repeat(1, 1, dif_pos.shape[-1])
        loss = 0
        for i in range(batch_size):
            sample_result = result[torch.where(batch == i)[0], :, :]
            pred = sample_result.sum(dim=(0, 1))
            target = dif_y[i]
            loss += torch.mean(torch.abs(pred - target) / (torch.abs(target) + eps))

        return loss / batch_size


class my_mae_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, g_pred, grafo):
        dif_pos, dif_y = grafo.dif_pos, grafo.dif_y
        batch_size = grafo.batch_size
        batch = grafo.batch

        result = dif_pos * g_pred.unsqueeze(-1).repeat(1, 1, dif_pos.shape[-1])
        loss = 0
        for i in range(batch_size):
            sample_result = result[torch.where(batch == i)[0], :, :]
            pred = sample_result.sum(dim=(0, 1))
            target = dif_y[i]
            loss += torch.mean(torch.abs(pred - target))

        return loss / batch_size
