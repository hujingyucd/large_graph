from typing import Tuple
import random
import torch
from sampler.base_sampler import Sampler
from torch_geometric.utils import subgraph


class MDRandomWalkSampler(Sampler):
    def __init__(self, rw_length: int = 10, node_budget: int = 1000):
        super(MDRandomWalkSampler, self).__init__()
        assert rw_length > 0, "rw_length {} < 0".format(rw_length)
        self.rw_length = rw_length
        self.node_budget = node_budget
        root_num = node_budget // rw_length
        assert 0 < root_num, (
            "node_budget {} smaller than rw_length {}".format(
                node_budget, rw_length))
        self.root_num = root_num

    def sample(self, edges: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        edges: 2*E
        return:
            new edges
            vs: length N bool tensor indicating new nodes
        """
        assert edges.size(0) == 2
        node_num = torch.max(edges) + 1
        vfs = torch.randint(0, node_num, (self.root_num, ))
        vs = torch.zeros(node_num, dtype=torch.bool)
        vs[vfs] = True
        for _ in range(node_num - self.root_num):
            degrees = ((edges[0] == u).sum().item() for u in vfs)
            uid = random.choices(range(self.root_num), degrees)[0]
            u = vfs[uid]
            neighbors = edges[1][edges[0] == u]
            v = neighbors[torch.randint(0, neighbors.size(0), (1, ))]
            vfs[uid] = v
            vs[v] = True
        new_edges, _ = subgraph(vs, edges, relabel_nodes=True)
        return new_edges, vs
