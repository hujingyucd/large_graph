from typing import Tuple
import torch


class Sampler():
    def __init__(self):
        pass

    def __call__(self,
                 edges: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        edges: 2*E
        return:
            new edges
            vs: length N bool tensor indicating new nodes
        """
        return self.sample(edges)

    def sample(self, edges: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        edges: 2*E
        return:
            new edges
            vs: length N bool tensor indicating new nodes
        """
        pass
