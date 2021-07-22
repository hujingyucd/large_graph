from abc import ABC, abstractmethod
from torch_geometric.data import Dataset
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
import random

# Helper function for visualization.
import torch
import networkx as nx
import matplotlib.pyplot as plt


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

        To create a subclass, you need to implement the following four functions:
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
    def solve(self):
        """
        For MIS Problem:
            Given the input, return the solutions(list of index of selected nodes, the size of the graph)
        """
        pass

    def metric(self):
        """
        TODO:For MIS Problem:
            Given the solutions, compute the evaluate result.(the size of the IS/the size of the graph)
        """
        # print("MIS: ", end="")
        # print(self.solution)
        print("Size of the IS: ", end="")
        print(len(self.solution))
        print(
            f'Given the solutions, compute the performance metrics:  {len(self.solution) / self.G.num_nodes:.2f}'
        )

        # m = torch.zeros([self.G.num_nodes])
        # for i in range(self.G.num_nodes):
        #     if i in self.solution:
        #         m[i] = torch.tensor(1)
        # G = to_networkx(self.G, to_undirected=True)
        # visualize(G, color=m)

    def eval(self, dataset: Dataset ,probs = None):
        """
        TODO:For MIS Problem:
            Given the input data, compute the corresponding solutions and evaluate result
        """
        # Get the first graph object as input graph.
        data = dataset[0]
        self.G = data
        self.probs = probs

        print("The input graph is:")
        print(data)
        print('=============================================================')
        print(f'Number of nodes: {data.num_nodes}')
        print(f'Number of edges: {data.num_edges}')
        print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        # print(f'Number of training nodes: {data.train_mask.sum()}')
        # print(
        #     f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}'
        # )
        # print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
        # print(f'Contains self-loops: {data.contains_self_loops()}')
        # print(f'Is undirected: {data.is_undirected()}')
        # print()
        '''
        # visualize the input graph
        G = to_networkx(data, to_undirected=True)
        m = torch.zeros([data.num_nodes])
        visualize(G, color=m)
        '''
        self.solve()
