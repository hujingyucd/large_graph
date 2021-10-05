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

    from solver.ml_solver import MLSolver
    from tiling.tile_graph import TileGraph
    from solver.ml_core.trainer import Trainer

    writer = SummaryWriter(log_dir=config["training"]["log_dir"])

    trainer = Trainer(
        gnn,
        dataset_train=["", None],
        dataset_test=["", None],
        device=device,
        model_save_path=config["training"]["model_save_path"],
        optimizer=None,
        writer=writer,
        logger_name="TRAINER",
        loss_weights=config["training"]["loss"],
        sample_per_epoch=config["training"]["sample_per_epoch"],
        sample_method=config["training"]["sample_method"],
        total_train_epoch=config["training"]["total_train_epoch"],
        save_model_per_epoch=config["training"]["save_model_per_epoch"],
        resume=False)

    solver = MLSolver(device=device, network=gnn, trainer=trainer)

    data_dict = torch.load(
        os.path.join(config["training"]["model_save_path"], "latest.pth"))
    epoch = data_dict["epoch"]
    solver.load_saved_network(
        os.path.join(config["training"]["model_save_path"],
                     "model_{}.pth".format(epoch)))

    complete_graph = TileGraph(config['tiling']['tile_count'])
    complete_graph.load_graph_state(config['tiling']['complete_graph_path'])

    from tiling.tile_solve import tile_silhouette_list

    tile_silhouette_list(
        solver=solver,
        complete_graph=complete_graph,
        silhouette_list=config["tiling"]["silhouette_list"],
        root_path=config["training"]["log_dir"])

    vdisplay.stop()
