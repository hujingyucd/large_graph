# from abc import ABC, abstractmethod
# from torch_geometric.data import Dataset
# from torch_geometric.datasets import KarateClub
# from torch_geometric.utils import to_networkx
# import random
import sys
import os

# Solvers for MIS problem
if __name__ == "__main__":

    currentdir = os.path.dirname(os.path.realpath(__file__))
    parentdir = os.path.dirname(currentdir)
    sys.path.append(parentdir)

    # import solver.MIS.base_solver
    from solver.MIS.greedy_solver import GreedySolver
    from solver.MIS.random_solver import RandomSolver
    from solver.ml_core.datasets import GraphDataset

    ''' Choose the dataset '''
    data_path = "/data/jingyu/graph/large_graph/data"
    dataset = GraphDataset(root=data_path, split="debug", subgraph_num=1)
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    # print(f'Number of classes: {dataset.num_classes}')
    print()

    print("Greedy: 0; Random: 1")
    flag = int(input("Select the algorithm: "))
    print()

    if flag == 0:
        print("Greedy solver!")
        print()
        T = GreedySolver()
        T.eval(dataset)
    else:
        print("Random solver!")
        print()
        R = RandomSolver()
        R.eval(dataset)
