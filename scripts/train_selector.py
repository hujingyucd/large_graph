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
    from networks.pseudo_tilingnn import PseudoTilinGNN as Gnn

    Path(config["selector"]["training"]["log_dir"]).mkdir(parents=True,
                                                          exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(config["selector"]["training"]["log_dir"],
                              "all.log"),
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
        level=getattr(logging, config["logging_level"]),
        datefmt='%Y-%m-%d %H:%M:%S')

    seed = config["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    rs = RandomState(MT19937(SeedSequence(seed)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gnn = Gnn(network_depth=config["solver"]["network"]["depth"],
              network_width=config["solver"]["network"]["width"],
              output_dim=config["solver"]["network"]["output_dim"]).to(device)

    trainer = Trainer(
        gnn,
        dataset_train=[None],
        dataset_test=[None],
        device=device,
        model_save_path=config["solver"]["training"]["model_save_path"],
        optimizer=torch.optim.Adam(
            gnn.parameters(),
            eps=1e-4,
            lr=config["solver"]["training"]["optimizer"]["lr"]),
        writer=SummaryWriter(log_dir=config["solver"]["training"]["log_dir"]),
        logger_name="TRAINER",
        loss_weights=config["solver"]["training"]["loss"],
        sample_per_epoch=config["solver"]["training"]["sample_per_epoch"],
        sample_method=config["solver"]["training"]["sample_method"],
        total_train_epoch=config["solver"]["training"]["total_train_epoch"],
        save_model_per_epoch=config["solver"]["training"]
        ["save_model_per_epoch"])

    from solver.ml_solver import MLSolver
    from tiling.tile_graph import TileGraph

    solver = MLSolver(device=device,
                      network=gnn,
                      trainer=trainer,
                      solve_method=config["solver"]["solve_method"])
    solver.network.eval()

    complete_graph = TileGraph(config['tiling']['tile_count'])
    complete_graph.load_graph_state(config['tiling']['complete_graph_path'])

    from selector.trainer import SelectorTrainer
    from selector.datasets import SampleGraphDataset
    from sampler.random_walk_sampler import RandomWalkSampler

    dataset_train = SampleGraphDataset(
        root=config["selector"]["training"]["data_path"],
        complete_graph=complete_graph,
        device=device,
        split='train',
        subgraph_num=config["selector"]["training"]["dataset_size"]["train"])
    dataset_test = SampleGraphDataset(
        root=config["selector"]["training"]["data_path"],
        complete_graph=complete_graph,
        device=device,
        split='test',
        subgraph_num=config["selector"]["training"]["dataset_size"]["train"])
    selector_net = Gnn(
        network_depth=config["selector"]["network"]["depth"],
        network_width=config["selector"]["network"]["width"],
        output_dim=config["selector"]["network"]["output_dim"]).to(device)

    fh = logging.FileHandler(
        os.path.join(config["selector"]["training"]["log_dir"],
                     "training.log"))
    fh.setFormatter(
        logging.Formatter('%(asctime)s %(name)s %(levelname)-8s %(message)s'))
    logging.getLogger("TRAINER").addHandler(fh)
    selector_trainer = SelectorTrainer(
        selector_net,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        device=device,
        model_save_path=config["selector"]["training"]["model_save_path"],
        optimizer=torch.optim.Adam(
            gnn.parameters(),
            eps=1e-4,
            lr=config["selector"]["training"]["optimizer"]["lr"]),
        sampler=RandomWalkSampler(rw_length=10, node_budget=300),
        solver=solver,
        writer=SummaryWriter(
            log_dir=config["selector"]["training"]["log_dir"]),
        logger_name="TRAINER",
        total_train_epoch=config["selector"]["training"]["total_train_epoch"],
        save_model_per_epoch=config["selector"]["training"]
        ["save_model_per_epoch"])

    if config["selector"]["training"]["show_intermediate"]:
        from interfaces.qt_plot import Plotter
        selector_trainer.train(plotter=Plotter())
    else:
        selector_trainer.train(plotter=None)
    vdisplay.stop()
