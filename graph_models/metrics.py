import torch
from torchmetrics import Metric

class MyMAE_Metric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_graphs", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, g_pred, grafo):
        dif_pos = grafo.dif_pos
        dif_y = grafo.dif_y
        batch = grafo.batch

        prod = torch.einsum('nd,ndm->nm', g_pred, dif_pos)
        batch_size = batch.max().item() + 1

        prod_sum = torch.zeros(batch_size, prod.shape[1], device=prod.device)
        prod_sum.index_add_(0, batch, prod)

        mae_per_graph = torch.mean(torch.abs(prod_sum - dif_y), dim=1)
        self.sum_abs_error += mae_per_graph.sum()
        self.total_graphs += mae_per_graph.numel()

    def compute(self):
        return self.sum_abs_error / self.total_graphs


class MyMAPE_Metric(Metric):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.add_state("sum_mape", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_graphs", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, g_pred, grafo):
        dif_pos = grafo.dif_pos
        dif_y = grafo.dif_y
        batch = grafo.batch

        prod = torch.einsum('nd,ndm->nm', g_pred, dif_pos)
        batch_size = batch.max().item() + 1

        pred_sum = torch.zeros(batch_size, prod.shape[1], device=prod.device)
        pred_sum.index_add_(0, batch, prod)

        mask = dif_y != 0
        abs_error = torch.abs((pred_sum - dif_y) / (torch.abs(dif_y) + self.eps))
        masked_error = abs_error * mask
        count_nonzero = mask.sum(dim=1)

        mape_per_graph = torch.where(
            count_nonzero > 0,
            masked_error.sum(dim=1) / count_nonzero,
            torch.tensor(0.0, device=abs_error.device)
        )

        self.sum_mape += mape_per_graph.sum()
        self.total_graphs += (count_nonzero > 0).sum()

    def compute(self):
        return self.sum_mape / self.total_graphs
