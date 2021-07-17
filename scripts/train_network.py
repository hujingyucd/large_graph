import logging
import torch
import sys
import os

if __name__ == "__main__":
    currentdir = os.path.dirname(os.path.realpath(__file__))
    parentdir = os.path.dirname(currentdir)
    sys.path[0] = parentdir

    from solver.ml_core.trainer import Trainer
    from solver.ml_core.datasets import GraphDataset
    from networks.pseudo_tilingnn import PseudoTilinGNN as Gnn

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn = Gnn().to(device)

    data_path = "/research/dept8/fyp21/cwf2101/data/brightkite"
    dataset_train = GraphDataset(root=data_path, split="debug", subgraph_num=1)
    dataset_test = dataset_train
    optimizer = torch.optim.Adam(gnn.parameters(), lr=1e-3)
    trainer = Trainer(gnn,
                      dataset_train=dataset_train,
                      dataset_test=dataset_test,
                      device=device,
                      model_save_path="./released_models",
                      optimizer=optimizer)
    trainer.train()
