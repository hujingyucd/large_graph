import logging
import torch
import sys
import os
import random
import argparse
import json
from pathlib import Path
from xvfbwrapper import Xvfb
from torch.utils.tensorboard import SummaryWriter
from numpy.random import MT19937, RandomState, SeedSequence

if __name__ == "__main__":

    currentdir = os.path.dirname(os.path.realpath(__file__))
    parentdir = os.path.dirname(currentdir)
    sys.path.append(parentdir)

    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--config",
                        type=str,
                        default=parentdir + "/configs/config.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    currentdir = os.path.dirname(os.path.realpath(__file__))
    parentdir = os.path.dirname(currentdir)
    sys.path[0] = parentdir

    vdisplay = Xvfb()
    vdisplay.start()

    from solver.ml_core.trainer import Trainer
    from solver.ml_core.datasets import TileGraphDataset
    from networks.pseudo_tilingnn import PseudoTilinGNN as Gnn

    Path(config["training"]["log_dir"]).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(config["training"]["log_dir"], "output.log"),
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
        level=getattr(logging, config["training"]["logging_level"]),
        datefmt='%Y-%m-%d %H:%M:%S')

    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    rs = RandomState(MT19937(SeedSequence(seed)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gnn = Gnn(network_depth=config["network"]["depth"],
              network_width=config["network"]["width"],
              output_dim=config["network"]["output_dim"]).to(device)
    # import torch_geometric.nn.data_parallel as data_parallel
    # import torch.nn as nn
    # gnn = data_parallel(gnn,device_ids = [0,2,3])
    # gnn = nn.DataParallel(gnn,device_ids=[0,2,3])

    data_path = config["training"]["data_path"]
    dataset_train = TileGraphDataset(root=data_path,
                                     split="train",
                                     subgraph_num=4000)
    dataset_test = TileGraphDataset(root=data_path,
                                    split="test",
                                    subgraph_num=800)

    optimizer = torch.optim.Adam(gnn.parameters(),
                                 eps=1e-4,
                                 lr=config["training"]["optimizer"]["lr"])

    writer = SummaryWriter(log_dir=config["training"]["log_dir"])

    trainer_logger = logging.getLogger("trainer")

    trainer = Trainer(
        gnn,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        device=device,
        model_save_path=config["training"]["model_save_path"],
        optimizer=optimizer,
        writer=writer,
        logger_name="TRAINER",
        loss_weights=config["training"]["loss"],
        sample_per_epoch=config["training"]["sample_per_epoch"],
        sample_method=config["training"]["sample_method"],
        total_train_epoch=config["training"]["total_train_epoch"],
        save_model_per_epoch=config["training"]["save_model_per_epoch"])

    from solver.ml_solver import MLSolver
    from tiling.tile_graph import TileGraph
    from tiling.tile_solve import tile_silhouette_list

    solver = MLSolver(device=device, network=gnn, trainer=trainer)

    complete_graph = TileGraph(config['tiling']['tile_count'])
    complete_graph.load_graph_state(config['tiling']['complete_graph_path'])
    tile_silhouette_list(solver,
                         complete_graph,
                         silhouette_list="./configs/silhouette_list.txt",
                         root_path="./results/output")

    # trainer.train(plotter=Plotter(),
    #               solver=solver,
    #               complete_graph=complete_graph)

    vdisplay.stop()
