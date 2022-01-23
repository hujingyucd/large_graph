from typing import Union
import numpy as np
import torch

from tiling.brick_layout import BrickLayout


def coverage_score(predict: Union[torch.Tensor, np.ndarray],
                   brick_layout: BrickLayout,
                   device: torch.device,
                   EPS: float = 1e-7):
    if isinstance(predict, np.ndarray):
        predict = torch.from_numpy(np.array(predict)).float().to(device)
    elif isinstance(predict, torch.Tensor):
        predict = predict.float().to(device)
    else:
        raise TypeError()
    x, _, _, _, _ = brick_layout.get_data_as_torch_tensor(device)

    # calculate total area
    filled_area = predict.dot(x[:, -1] * brick_layout.complete_graph.max_area
                              ) / brick_layout.get_super_contour_poly().area
    try:
        assert filled_area >= -EPS and filled_area <= 1 + EPS
    except AssertionError as e:
        print(torch.sum(predict))
        print(filled_area)
        print(predict.shape)
        print(brick_layout.complete_graph.max_area)
        print(brick_layout.get_super_contour_poly().area)
        print(x.shape)
        raise e

    return (filled_area).detach().cpu().item()
