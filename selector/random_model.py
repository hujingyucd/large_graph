import torch


class RandomPredictor(torch.nn.Module):
    def forward(self,
                x: torch.Tensor,
                col_e_idx: torch.Tensor,
                col_e_features: torch.Tensor = None):
        return torch.rand((x.size(0), 1))
