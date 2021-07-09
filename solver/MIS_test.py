from abc import ABC, abstractmethod
from torch_geometric.data import Dataset
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
import random

# Solvers for MIS problem
import base_solver
import greedy_solver
import random_solver

if __name__ == "__main__":
    ''' Choose the dataset '''
    dataset = KarateClub()
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print()

    print("Greedy: 0; Random: 1")
    flag = int(input("Select the algorithm: "))
    print()

    if flag == 0:
        print("Greedy solver!")
        print()
        T = greedy_solver.GreedySolver()
        T.eval(dataset)
    else:
        print("Random solver!")
        print()
        R = random_solver.RandomSolver()
        R.eval(dataset)
