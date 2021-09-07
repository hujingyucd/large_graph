import torch
from copy import deepcopy

from solver.MIS.base_solver import BaseSolver
from solver.ml_core.trainer import Trainer
from tiling.brick_layout import BrickLayout


class MLSolver(BaseSolver):
    def __init__(self,
                 device,
                 network,
                 num_prob_maps=1,
                 trainer: Trainer = None):
        self.device = device
        # self.complete_graph = complete_graph
        self.network = network
        self.num_prob_maps = num_prob_maps
        self.trainer: Trainer = trainer
        self.trainer.solver = self

    def load_saved_network(self, network_path):
        self.network.load_state_dict(
            torch.load(network_path, map_location=self.device))

    def greedy(self, graph):
        # base cases
        if (len(graph) == 0):
            return []

        if (len(graph) == 1):
            return [list(graph.keys())[0]]

        # Greedily select a vertex from the graph
        current = list(graph.keys())[0]

        # Case 1 - current node not included
        graph2 = dict(graph)
        del graph2[current]

        res1 = self.greedy(graph2)

        # Case 2 - current node included
        # Delete its neighbors
        for v in graph[current]:
            if (v in graph2):
                del graph2[v]

        res2 = [current] + self.greedy(graph2)

        # select the maximum set
        if (len(res1) > len(res2)):
            return res1
        return res2

    def predict(self, brick_layout: BrickLayout):
        predictions = None
        if len(brick_layout.collide_edge_index) == 0 or len(
                brick_layout.align_edge_index) == 0:
            # only collision edges left, select them all
            predictions = torch.ones(
                (brick_layout.node_feature.shape[0],
                 self.num_prob_maps)).float().to(self.device)
        else:
            # convert to torch tensor
            x, _, _, coll_e_index, _ = brick_layout.get_data_as_torch_tensor(
                self.device)

            # get network prediction
            predictions: torch.Tensor = self.network(x=x[:, -1].unsqueeze(-1),
                                                     col_e_idx=coll_e_index)

        # get the minimium loss map
        best_map_index = self.get_best_prob_map(predictions, brick_layout)
        selected_prob = predictions[:, best_map_index].detach().cpu().numpy()

        return selected_prob

    def solve(self, brick_layout=None) -> (BrickLayout, float):
        solution, score, predict_order = self._solve_by_probablistic_greedy(
            brick_layout)
        output_layout = deepcopy(brick_layout)
        output_layout.predict_order = predict_order
        output_layout.predict = solution
        output_layout.predict_probs = self.predict(brick_layout)
        return output_layout, score

    def get_unsupervised_losses_from_layout(
            self, layout: BrickLayout, probs: torch.Tensor) -> torch.Tensor:
        """
        probs: of shape N * M, N node num, M prob map num
        return: of length M
        currently M has to be 1 due to loss implementation
        """
        x, _, _, collide_edge_index, _ = layout.get_data_as_torch_tensor(
            self.device)
        area = x[:, -1].unsqueeze(-1)

        return self.trainer.calculate_unsupervised_losses(
            probs, area, collide_edge_index)

    def get_best_prob_map(self,
                          probs: torch.Tensor,
                          layout: BrickLayout,
                          top_k_num: int = 1) -> torch.LongTensor:
        """
        probs: of shape N * M, N node num, M prob map num
        return: scalar, or of length top_k_num
        currently M has to be 1 due to loss implementation
        """
        losses = self.get_unsupervised_losses_from_layout(layout, probs)
        selected_map = torch.argsort(losses)[:top_k_num]
        if top_k_num == 1:
            selected_map.squeeze_()
        return selected_map
