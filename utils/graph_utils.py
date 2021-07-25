import torch
from torch_geometric.data import Data


def sample_solution_greedy(graph: Data,
                           priority: torch.Tensor) -> torch.LongTensor:
    """
    Greedily generate maximal independent set with given priority.
    graph: torch_geometric Data object
    weight: N * 1 tensor
    output: length N tensor
    """
    _, indices = torch.sort(priority, dim=0, descending=True)
    indices.squeeze_()

    output = torch.full_like(indices, fill_value=False, dtype=torch.bool)
    mark = torch.full_like(indices, fill_value=False, dtype=torch.bool)

    adj_list = [[] for _ in indices]
    for i, j in torch.transpose(graph.edge_index, 0, 1).tolist():
        adj_list[i].append(j)
        adj_list[j].append(i)

    for node_idx in indices:
        if not mark[node_idx]:
            output[node_idx] = True
            mark[node_idx] = True
            mark[adj_list[node_idx]] = True

    return torch.arange(0, len(indices), dtype=torch.long)[output]
