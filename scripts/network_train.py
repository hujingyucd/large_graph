import logging
import sys
import os
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    currentdir = os.path.dirname(os.path.realpath(__file__))
    parentdir = os.path.dirname(currentdir)
    sys.path[0] = parentdir

    from solver.trainer import Trainer
    from networks.pseudo_tilingnn import PseudoTilinGNN

    dataset_path = "/research/dept8/fyp21/cwf2101/data/brightkite",
    model_save_path = "./released_models",
    batch_size = 1
    learning_rate = 1e-3
    training_epoch = 10000
    save_model_per_epoch = 2

    #### Network
    network = PseudoTilinGNN().to(device)

    ## solver
    # ml_solver = ML_Solver(debugger, device, data_env.complete_graph, network, num_prob_maps=1)

    #### Optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    trainer = Trainer(device, 
                      network, 
                      optimizer=optimizer, 
                      data_path="/research/dept8/fyp21/cwf2101/data/brightkite", 
                      model_save_path = "./released_models")

    trainer.train(optimizer          = optimizer,
                  batch_size         = batch_size,
                  training_epoch     = training_epoch,
                  save_model_per_epoch = save_model_per_epoch)

