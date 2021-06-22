from abc import ABC, abstractmethod
from torch_geometric.data import Dataset


class BaseSolver(ABC):
    """This class is an abstract base class(ABC) for Solvers

        To create a subclass, you need to implement the following four functions:
        --<__init__>: initilize the class,
        --<eval>: Given the solutions, output the algorithm's performance
        --<solve>: Given the input data and algorithms, output the solutions
    """
    def __init__(self):
        return

    @abstractmethod
    def solve(self):
    """
        For MIS Problem:
            Given the input, return the solutions(list of index of selected nodes, the size of the graph)
    """
        return

    @abstractmethod
    def metric(self):
    """
        For MIS Problem:
            Given the solutions, compute the evaluate result.(the size of the IS/the size of the graph)
    """

        return

    @abstractmethod
    def eval(self, dataset : Dataset):
    """
        For MIS Problem:
            Given the input data, compute the corresponding solutions and evaluate result
    """

        return



