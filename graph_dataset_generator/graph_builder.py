import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, to_edge_index
import numpy as np

def edge_index_gen(adj,rate):
    """
    Generates `edge_index` and `edge_weight` tensors for PyTorch Geometric 
    from a binary adjacency matrix and a matrix of link rates.

    Parameters:
    - adj (torch.Tensor): Square binary adjacency matrix where adj[i, j] = 1 indicates a link between agents i and j.
    - rate (torch.Tensor): Square matrix of the same shape as `adj`, containing the channel rate for each pair of agents.
      Only values where adj[i, j] == 1 are used.

    Returns:
    - edge_index (LongTensor): Edge indices (2 x num_edges) in COO format.
    - edge_weight (Tensor): Edge weights corresponding to the `rate` values for each active link.

    Notes:
    - Assumes the graph is undirected and that both `adj` and `rate` are symmetric.
    - The resulting graph will have identical weights in both directions for each edge.
    """
    adjacency_matrix =adj*rate
    edge_index, edge_weight= to_edge_index(adjacency_matrix.to_sparse())
    return edge_index, edge_weight

def points_to_graph(TA, NA, canal, MNF=None, edge_weight=True):
    """
    Constructs a PyTorch Geometric Data object representing a communication graph 
    given the positions of task and communication agents and a channel model.

    Parameters:
    - TA (torch.Tensor): Positions of task agents, shape (num_task_agents, 2).
    - NA (torch.Tensor): Positions of communication agents, shape (num_comm_agents, 2).
    - canal (object): Channel model object with `.adjacency()` and `.predict()` methods.
      - `canal.adjacency(pos)` returns a binary adjacency matrix.
      - `canal.predict(pos)` returns (rate_matrix, additional_outputs).
    - mfr (optional, any): Label or optimization result (e.g., max flow rate) to assign to the graph. Default is None.
    - edge_weight (bool): Whether to include edge weights (link rates) in the `Data` object.

    Returns:
    - data (torch_geometric.data.Data): Graph data object containing:
        - `x`: One-hot encoded agent types: [1,0] for task agents, [0,1] for communication agents.
        - `edge_index`: Edge indices in COO format.
        - `edge_attr` (optional): Edge weights (channel rates) if `edge_weight=True`.
        - `pos`: Positions of all agents (task + communication).
        - `y` (optional): Max flow rate or optimization result if `mfr` is provided.
    """

    # Combine agent positions
    positions = torch.vstack((TA, NA))
    task_agents = TA.shape[0]
    comm_agents = NA.shape[0]

    # One-hot encode agent types: [1,0] for tasks, [0,1] for comms
    types = torch.zeros((task_agents + comm_agents, 2))
    types[:task_agents, 0] = 1
    types[task_agents:, 1] = 1

    # Generate adjacency and link rate matrices
    pos_np = positions.detach().numpy()
    adj = canal.adjacency(pos_np)
    rate, _ = canal.predict(pos_np)

    # Build edge_index and edge weights from adjacency and rate
    edge_index, edge_weights = edge_index_gen(adj=torch.from_numpy(adj), rate=torch.from_numpy(rate))
    #edge_index, edge_weights = to_undirected(edge_index, edge_attr=edge_weights)

    # Assemble Data object
    data_kwargs = {
        "x": types,
        "edge_index": edge_index,
        "pos": positions
    }

    if edge_weight:
        data_kwargs["edge_attr"] = edge_weights

    if MNF is not None:
        data_kwargs["y"] = MNF

    return Data(**data_kwargs)
