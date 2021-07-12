
import os
from solver.trainer import Trainer
import torch
import numpy as np
from networks.pseudo_tilingnn import PseudoTilinGNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    dataset_path = os.path.join('./dataset', f"{env_name}-ring{complete_graph_size}-{number_of_data}")
    batch_size = 1
    learning_rate = 1e-3
    training_epoch = 10000
    save_model_per_epoch = 2

    #### Network
    network = PseudoTilinGNN().to(device)

    ## solver
    # ml_solver = ML_Solver(debugger, device, data_env.complete_graph, network, num_prob_maps=1)

    #### Optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    trainer = Trainer(device, network, data_path=dataset_path)

    trainer.train(ml_solver          = None,
                  optimizer          = optimizer,
                  batch_size         = batch_size,
                  training_epoch     = training_epoch,
                  save_model_per_epoch = save_model_per_epoch)

