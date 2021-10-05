from typing import Tuple
from abc import ABC, abstractmethod
from copy import deepcopy
from torch_geometric.data import Dataset
import torch
import networkx as nx
import matplotlib.pyplot as plt

from tiling.brick_layout import BrickLayout


def visualize(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        nx.draw_networkx(h,
                         pos=nx.spring_layout(h, seed=42),
                         with_labels=False,
                         node_color=color,
                         cmap="Set2")
    plt.show()


'''======================= Base solver =========================='''


class BaseSolver(ABC):
    """This class is an abstract base class(ABC) for Solvers

        To create a subclass,
        you need to implement the following four functions:
        --<__init__>: initilize the class,
        --<eval>: Given the solutions, output the algorithm's performance
        --<solve>: Given the input data and algorithms, output the solutions
        --<metric>: Given the solutions, compute the performance metrics
    """
    def __init__(self):
        self.G = None
        self.solution = None
        self.probs = None

    @abstractmethod
    def predict(self, brick_layout: BrickLayout) -> (torch.Tensor):
        pass

    @abstractmethod
    def solve(self, brick_layout: BrickLayout) -> Tuple[BrickLayout, float]:
        """
        For MIS Problem:
            Given the input, return the solutions
            (list of index of selected nodes, the size of the graph)
        """
        pass

    def metric(self):
        """
        TODO:For MIS Problem:
            Given the solutions, compute the evaluate result.
            (the size of the IS/the size of the graph)
        """
        # print("MIS: ", end="")
        # print(self.solution)
        print("Size of the IS: ", end="")
        print(len(self.solution))
        print('Given the solutions, compute the performance metrics: ' +
              f'{len(self.solution) / self.G.num_nodes:.2f}')

        # m = torch.zeros([self.G.num_nodes])
        # for i in range(self.G.num_nodes):
        #     if i in self.solution:
        #         m[i] = torch.tensor(1)
        # G = to_networkx(self.G, to_undirected=True)
        # visualize(G, color=m)

    def eval(self, dataset: Dataset, probs=None):
        """
        TODO:For MIS Problem:
            Given the input data,
            compute the corresponding solutions and evaluate result
        """
        # # Get the first graph object as input graph.
        # data = dataset[0]
        # self.G = data
        # self.probs = probs

        # print("The input graph is:")
        # print(data)
        # print('=============================================================')
        # print(f'Number of nodes: {data.num_nodes}')
        # print(f'Number of edges: {data.num_edges}')
        # print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        # '''
        # # visualize the input graph
        # G = to_networkx(data, to_undirected=True)
        # m = torch.zeros([data.num_nodes])
        # visualize(G, color=m)
        # '''
        # self.solve()

    def solve_with_trials(self, brick_layout: BrickLayout,
                          trial_times: int) -> Tuple[BrickLayout, float]:
        current_max_score = 0.0
        current_best_solution = None

        for i in range(trial_times):
            result_brick_layout, score = self.solve(brick_layout)

            if score != 0:
                if current_max_score < score:
                    current_max_score = score
                    current_best_solution = deepcopy(result_brick_layout)
                    print(f"current_max_score : {current_max_score}")

        assert current_best_solution is not None
        return current_best_solution, current_max_score
