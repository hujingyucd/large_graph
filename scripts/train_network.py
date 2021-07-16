import logging
import torch
import sys
import os

if __name__ == "__main__":
    currentdir = os.path.dirname(os.path.realpath(__file__))
    parentdir = os.path.dirname(currentdir)
    sys.path[0] = parentdir

    from solver.ml_core.trainer import Trainer
    from networks.pseudo_tilingnn import PseudoTilinGNN as Gnn

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn = Gnn().to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=1e-3)
    trainer = Trainer(
        gnn,
        data_path="/research/dept8/fyp21/cwf2101/data/brightkite",
        device=device,
        model_save_path="./released_models",
        optimizer=optimizer)
    trainer.train()
