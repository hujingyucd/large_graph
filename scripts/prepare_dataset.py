import os
import sys

import argparse
import logging

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data/temp")
    parser.add_argument("--url", type=str, default="no")
    args = parser.parse_args()

    currentdir = os.path.dirname(os.path.realpath(__file__))
    parentdir = os.path.dirname(currentdir)
    sys.path.append(parentdir)
    # from solver.ml_core.datasets import GraphDataset
    # ds = GraphDataset(root=args.root, url=args.url, split="train")
    # ds = GraphDataset(root=args.root,
    #                   url=args.url,
    #                   split="test",
    #                   subgraph_num=2000)
    from solver.ml_core.datasets import TileGraphDataset
    ds = TileGraphDataset(root=args.root, split="train", subgraph_num=2000)
    ds = TileGraphDataset(root=args.root, split="test", subgraph_num=400)
