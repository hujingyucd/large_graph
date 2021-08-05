import os
import sys

import pickle
import argparse
import logging

import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data/tilin-graph")
    parser.add_argument(
        "--path",
        type=str,
        default=("/research/dept8/fyp21/cwf2101/data/temp.pkl"))
    args = parser.parse_args()

    currentdir = os.path.dirname(os.path.realpath(__file__))
    parentdir = os.path.dirname(currentdir)
    sys.path.append(parentdir)
    from solver.ml_core.datasets import GraphDataset

    with open(args.path, "rb") as f:
        data = pickle.load(f)

    edge_index = data["colli_edges"]
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index.transpose_(0, 1)
    nodes = torch.unique(edge_index)
    logging.info("unique")
    edge_index, _ = subgraph(nodes, edge_index, relabel_nodes=True)

    data = Data(edge_index=edge_index)
    data.num_nodes = data.num_nodes

    os.mkdir(os.path.join(args.root, "raw"))
    torch.save(data, os.path.join(args.root, "raw", 'complete_graph.pt'))

    ds = GraphDataset(root=args.root,
                      url="no",
                      split="train",
                      subgraph_num=500)
    ds = GraphDataset(root=args.root, url="no", split="test", subgraph_num=200)
