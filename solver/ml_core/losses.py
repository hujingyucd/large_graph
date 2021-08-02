import torch
import torch.nn as nn


class AreaLoss(nn.Module):
    def __init__(self, weight=1.0, eps=1e-7):
        super(AreaLoss, self).__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, x, areas=None):
        ''' x: N * 1 tensor
            areas: normalized area, N * 1 tensor
        '''
        if areas is None:
            avg_area = torch.mean(x)
        else:
            # different from tilingnn, to be confirmed
            avg_area = x * areas / torch.sum(areas)
        avg_area = torch.clamp(avg_area, min=self.eps)
        result = 1.0 - self.weight * torch.log(avg_area)
        # assert 1.0 <= result
        return result


class OverlapLoss(nn.Module):
    def __init__(self, weight=10.0, eps=1e-7):
        super(OverlapLoss, self).__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, x, edge_index):
        ''' x: N * 1 tensor
            edge_index: 2 * |E_{ovl}|
        '''
        x = torch.squeeze(x)
        if len(edge_index) == 0:
            return 1.0
        prob_i = torch.gather(x, dim=0, index=edge_index[0])
        prob_k = torch.gather(x, dim=0, index=edge_index[1])

        result = 1.0 - self.weight * torch.mean(
            torch.log(1 - torch.clamp(
                prob_i * prob_k, min=self.eps, max=1 - self.eps)))
        # assert 1.0 <= result
        return result


class SolutionLoss(nn.Module):
    def __init__(self, weight=10.0, eps=1e-7):
        super(SolutionLoss, self).__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, x, solution_mask, reward):
        result = 1 - self.weight * torch.mean(
            torch.where(solution_mask, 1.0, -1.0) * torch.log(x)) * reward
        return result
