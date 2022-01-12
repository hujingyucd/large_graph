from typing import Tuple, Literal
import torch
from copy import deepcopy
import numpy as np

from utils.solver_util import SelectionSolution
from utils.solver_util import label_collision_neighbor, create_solution

from solver.MIS.base_solver import BaseSolver
from solver.ml_core.trainer import Trainer
from tiling.brick_layout import BrickLayout


class MLSolver(BaseSolver):

    def __init__(self,
                 device: torch.device,
                 network: torch.nn.Module,
                 trainer: Trainer,
                 solve_method: Literal["probablistic",
                                       "onetime"] = "probablistic",
                 num_prob_maps: int = 1):
        self.device = device
        # self.complete_graph = complete_graph
        self.network = network
        self.num_prob_maps = num_prob_maps
        self.trainer = trainer
        self.solve_method = solve_method
        if trainer:
            self.trainer.solver = self

    def load_saved_network(self, network_path):
        self.network.load_state_dict(
            torch.load(network_path, map_location=self.device))

    def _solve_by_probablistic_greedy(self, origin_layout: BrickLayout):

        node_num = origin_layout.node_feature.shape[0]
        collision_edges = origin_layout.collide_edge_index

        # Initial the variables
        current_solution = SelectionSolution(node_num)
        round_cnt = 1

        while len(current_solution.unlabelled_nodes) > 0:

            # create layout for currently unselected nodes
            temp_layout, node_re_index = origin_layout.compute_sub_layout(
                current_solution)
            prob = self.predict(temp_layout)

            # compute new probs_value
            previous_prob = np.array(
                list(current_solution.unlabelled_nodes.values()))
            prob_per_node = np.power(
                np.power(previous_prob, round_cnt - 1) * prob, 1 / round_cnt)

            # update the prob saved
            for i in range(len(prob_per_node)):
                current_solution.unlabelled_nodes[
                    node_re_index[i]] = prob_per_node[i]  # update the prob

            # argsort the prob in descending
            sorted_indices = np.argsort(-prob_per_node)

            for idx in sorted_indices:
                origin_idx = node_re_index[idx]

                # collision handling
                if origin_idx not in current_solution.unlabelled_nodes:
                    break

                # conditional adding the node
                var = np.random.uniform()
                if np.exp((prob_per_node[idx] - 1) * 1.0) > var:
                    current_solution.label_node(origin_idx, 1, origin_layout)
                    current_solution = label_collision_neighbor(
                        collision_edges, current_solution, origin_idx,
                        origin_layout)

            # update the count
            round_cnt += 1

        # create bricklayout with prediction
        score, selection_predict, predict_order = create_solution(
            current_solution, origin_layout, device=self.device)

        return selection_predict, score, predict_order

    def _solve_by_one_time_greedy(self, origin_layout: BrickLayout):

        node_num = origin_layout.node_feature.shape[0]
        collision_edges = origin_layout.collide_edge_index

        # Initial the variables
        current_solution = SelectionSolution(node_num)

        # create layout for unselected nodes
        temp_layout, node_re_index = origin_layout.compute_sub_layout(
            current_solution)
        prob = self.predict(temp_layout)

        prob_per_node = prob

        # update the prob saved
        for i in range(len(prob_per_node)):
            current_solution.unlabelled_nodes[
                node_re_index[i]] = prob_per_node[i]

        # argsort the prob in descending
        sorted_indices = np.argsort(-prob_per_node)

        for idx in sorted_indices:
            origin_idx = node_re_index[idx]

            # collision handling
            if origin_idx not in current_solution.unlabelled_nodes:
                continue

            current_solution.label_node(origin_idx, 1, origin_layout)
            current_solution = label_collision_neighbor(
                collision_edges, current_solution, origin_idx, origin_layout)

        # create bricklayout with prediction
        score, selection_predict, predict_order = create_solution(
            current_solution, origin_layout, device=self.device)

        return selection_predict, score, predict_order

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
        predictions: torch.Tensor
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
            predictions = self.network(x=x[:, -1].unsqueeze(-1),
                                       col_e_idx=coll_e_index)

        # get the minimium loss map
        best_map_index = self.get_best_prob_map(predictions, brick_layout)
        selected_prob = predictions[:, best_map_index].detach().cpu().numpy()

        return selected_prob

    def solve(
        self,
        brick_layout=None,
    ) -> Tuple[BrickLayout, float]:
        # self.network.eval()
        if self.solve_method == "probablistic":
            (solution, score,
             predict_order) = self._solve_by_probablistic_greedy(brick_layout)
        elif self.solve_method == "onetime":
            (solution, score,
             predict_order) = self._solve_by_one_time_greedy(brick_layout)
        else:
            raise NotImplementedError("unknown solve method")
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
        if self.trainer is None:
            raise RuntimeError()
        x, _, _, collide_edge_index, _ = layout.get_data_as_torch_tensor(
            self.device)
        area = x[:, -1].unsqueeze(-1)

        return self.trainer.calculate_unsupervised_losses(
            probs, area, collide_edge_index)

    def get_best_prob_map(self,
                          probs: torch.Tensor,
                          layout: BrickLayout,
                          top_k_num: int = 1):
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
