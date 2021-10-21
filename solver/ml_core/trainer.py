import os
import sys
import logging
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from solver.ml_core.losses import AreaLoss, OverlapLoss, SolutionLoss
from solver.MIS.base_solver import BaseSolver
from utils.data_util import load_bricklayout
from tiling.tile_graph import TileGraph
from interfaces.qt_plot import Plotter


class Trainer():
    def __init__(self,
                 network,
                 dataset_train,
                 dataset_test,
                 device,
                 model_save_path,
                 optimizer,
                 writer: SummaryWriter,
                 logger_name="trainer",
                 loss_weights=None,
                 sample_per_epoch=0,
                 sample_method="bernoulli",
                 bernoulli_prob=0.81,
                 resume=True,
                 total_train_epoch=100,
                 save_model_per_epoch=5):

        self.device = device
        self.model_save_path = model_save_path

        self.network = network
        self.optimizer = optimizer

        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.loader_train = DataLoader(dataset_train,
                                       batch_size=1,
                                       shuffle=True)
        self.loader_test = DataLoader(dataset_test,
                                      batch_size=1,
                                      shuffle=False)

        if loss_weights:
            self.area_loss = AreaLoss(weight=loss_weights["area_weight"])
            self.collision_loss = OverlapLoss(
                weight=loss_weights["collision_weight"])
            self.solution_loss = SolutionLoss(
                weight=loss_weights["solution_weight"])
        else:
            self.area_loss = AreaLoss()
            self.collision_loss = OverlapLoss()
            self.solution_loss = SolutionLoss()
        # self.solution_loss = torch.nn.CrossEntropyLoss()
        if sample_method == "bernoulli":
            from utils.graph_utils import generate_bernoulli
            self.sample_solution = generate_bernoulli(prob=bernoulli_prob)
        elif sample_method == "greedy":
            from utils.graph_utils import sample_solution_greedy
            self.sample_solution = sample_solution_greedy
        elif sample_method == "solver":
            self.sample_solution = None
        self.sample_per_epoch = sample_per_epoch

        self.total_train_epoch = total_train_epoch
        self.save_model_per_epoch = save_model_per_epoch

        self.writer = writer
        self.logger = logging.getLogger(logger_name)

        self.epoch = 0
        self.min_test_loss = float("inf")
        self.min_test_loss_epoch = 0
        self.resume = resume

        if resume is True:
            self.load()

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, solver: BaseSolver):
        self._solver = solver

    def save(self, prefix=""):
        try:
            os.mkdir(self.model_save_path)
        except FileExistsError:
            pass
        torch.save(
            self.network.state_dict(),
            os.path.join(self.model_save_path,
                         "{}model_{}.pth".format(prefix, self.epoch)))
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(self.model_save_path,
                         "{}optimizer_{}.pth".format(prefix, self.epoch)))
        data_dict = {
            "epoch": self.epoch,
            "min_test_loss": self.min_test_loss,
            "min_test_loss_epoch": self.min_test_loss_epoch
        }

        torch.save(
            data_dict,
            os.path.join(self.model_save_path,
                         "{}datadict_{}.pth".format(prefix, self.epoch)))

        torch.save(
            data_dict,
            os.path.join(self.model_save_path, "{}latest.pth".format(prefix)))

    def load(self):
        target_path = os.path.join(self.model_save_path, "latest.pth")
        if not os.path.exists(target_path):
            return
        data_dict = torch.load(target_path)
        self.epoch = data_dict["epoch"]
        self.min_test_loss = data_dict["min_test_loss"]
        self.min_test_loss_epoch = data_dict["min_test_loss_epoch"]

        net_path = os.path.join(self.model_save_path,
                                "model_{}.pth".format(self.epoch))
        self.network.load_state_dict(torch.load(net_path))

        optim_path = os.path.join(self.model_save_path,
                                  "optimizer_{}.pth".format(self.epoch))
        self.optimizer.load_state_dict(torch.load(optim_path))

    def calculate_unsupervised_losses(
            self, probs: torch.Tensor, area: torch.Tensor,
            collide_edge_index: torch.Tensor) -> (torch.Tensor):
        """
        probs: of shape N * M, N node num, M prob map num
        area: of shape N * 1
        collide_edge_index: of shape 2 * E
        currently M has to be 1 due to loss implementation
        """
        result = self.area_loss(probs, area) * self.collision_loss(
            probs, collide_edge_index)
        return result.unsqueeze(0)

    def test_single_epoch(self, loader):
        # self.network.eval()
        area_losses = []
        collision_losses = []
        for batch in loader:
            torch.cuda.empty_cache()
            data = batch.to(self.device)
            probs = self.network(x=data.x, col_e_idx=data.edge_index)
            area_losses.append(self.area_loss(probs, data.x).detach())
            collision_losses.append(
                self.collision_loss(probs, data.edge_index).detach())
        area_losses = torch.stack(area_losses)
        collision_losses = torch.stack(collision_losses)
        losses = area_losses * collision_losses
        return torch.mean(losses), torch.mean(area_losses), torch.mean(
            collision_losses)

    def train_single_epoch(self,
                           plotter: Plotter = None,
                           solver: BaseSolver = None,
                           complete_graph: TileGraph = None):
        self.logger.info("training epoch start")
        torch.cuda.empty_cache()
        i = self.epoch
        train_metrics = {}
        self.network.train()
        for batch in self.loader_train:
            self.optimizer.zero_grad()
            # get prediction
            data = batch.to(self.device)
            probs = self.network(x=data.x, col_e_idx=data.edge_index)
            try:
                assert not torch.any(torch.isnan(probs))
            except AssertionError:
                torch.save(probs, "./probs.pt")
                self.save("nan")
                sys.exit("probs, data: {}".format(data.idx))

            loss_area = self.area_loss(probs, data.x)
            try:
                assert loss_area >= 1.0
            except AssertionError:
                self.logger.error("area loss: {}, data: {}".format(
                    loss_area, data.idx))
                self.save()

            loss_collision = self.collision_loss(probs, data.edge_index)
            try:
                assert loss_collision >= 1.0
            except AssertionError:
                self.logger.error("collision loss: {}, data: {}".format(
                    loss_collision, data.idx))
                self.save()

            train_loss = loss_area * loss_collision

            log_items = [
                f"{i}", "idx {}".format(data.idx.item()),
                "loss {:6f}".format(train_loss.item())
            ]

            if self.sample_per_epoch and i % self.sample_per_epoch == 0:
                if self.sample_solution:
                    with torch.no_grad():
                        _, mask = self.sample_solution(data, probs)
                        mask = torch.unsqueeze(mask, 1)
                        solution = torch.where(mask, 1.0, 0.0)
                        metric_area = self.area_loss(solution, data.x).detach()
                        metrics = metric_area * 1.0
                        reward = train_loss - metrics
                else:
                    processed_path = str(data.path[0])
                    queried_layout = self._load_raw_layout(
                        processed_path, complete_graph)
                    result_layout, _ = solver.solve(queried_layout)
                    with torch.no_grad():
                        metrics = 1.0
                        # metric for holes
                        num_holes = result_layout.detect_holes()
                        metric_hole = torch.log(
                            torch.tensor(float(num_holes)).to(
                                self.device)) * 0.1 + 1.0
                        metrics *= metric_hole
                        if "hole" not in train_metrics:
                            train_metrics["hole"] = []
                        train_metrics["hole"].append(metric_hole.item())

                        # metric for area
                        solution = torch.unsqueeze(
                            torch.from_numpy(
                                np.array(result_layout.predict_probs)).to(
                                    self.device), 1)
                        mask = torch.where(solution > 0.5, True, False)
                        metric_area = self.area_loss(solution, data.x).detach()
                        metrics *= metric_area
                        if "area" not in train_metrics:
                            train_metrics["area"] = []
                        train_metrics["area"].append(metric_area.item())

                        reward = train_loss - metrics

                loss_solution = self.solution_loss(probs, mask, reward)
                log_items += [
                    "metrics {:6f}".format(metrics.item()),
                    "loss_solution {:6f}".format(loss_solution.item()),
                    "metric_area {:6f}".format(metric_area.item())
                ]
                train_loss = train_loss * loss_solution

            try:
                train_loss.backward()
            except RuntimeError:
                self.save("backward")
            self.optimizer.step()

            self.logger.debug(", ".join(log_items))

        for key, vs in train_metrics.items():
            self.writer.add_scalar("Metric_{}/train".format(key),
                                   sum(vs) / len(vs), self.epoch)

        self.logger.info("training epoch done\n\n")

        self.logger.info("testing epoch start")
        torch.cuda.empty_cache()
        with torch.no_grad():
            loss_train, loss_train_area, loss_train_coll = \
                    self.test_single_epoch(self.loader_train)
        self.writer.add_scalar("Loss/train", loss_train, self.epoch)
        self.writer.add_scalar("AreaLoss/train", loss_train_area, self.epoch)
        self.writer.add_scalar("CollisionLoss/train", loss_train_coll,
                               self.epoch)

        torch.cuda.empty_cache()
        with torch.no_grad():
            loss_test, loss_test_area, loss_test_coll = self.test_single_epoch(
                self.loader_test)
        self.writer.add_scalar("Loss/test", loss_test, self.epoch)
        self.writer.add_scalar("AreaLoss/test", loss_test_area, self.epoch)
        self.writer.add_scalar("CollisionLoss/test", loss_test_coll,
                               self.epoch)

        self.logger.info("epoch {} testing loss {:.6f}".format(i, loss_test))
        self.logger.info("epoch {} testing area loss {:.6f}".format(
            i, loss_test_area))
        self.logger.info("epoch {} testing collision loss {:.6f}".format(
            i, loss_test_coll))
        self.logger.info("testing epoch end!\n\n")

        # result debugging
        if (loss_test < self.min_test_loss
                or i % self.save_model_per_epoch == 0):
            if loss_test < self.min_test_loss:
                self.min_test_loss = loss_test
                self.min_test_loss_epoch = self.epoch
            torch.cuda.empty_cache()
            self.save()
            self.logger.info("model saved at epoch {}".format(i))

    def _load_raw_layout(self, processed_path, complete_graph):
        raw_path = os.path.splitext("raw".join(
            processed_path.rsplit('processed', 1)))[0] + '.pkl'
        assert complete_graph is not None
        return load_bricklayout(raw_path, complete_graph)

    def evaluate(self,
                 plotter: Plotter = None,
                 complete_graph: TileGraph = None,
                 solver: BaseSolver = None,
                 split: str = "train",
                 idx: int = 0):
        if not all([plotter, solver, complete_graph]):
            self.logger.warning("no solving and visualization")
            return
        data = getattr(self, "dataset_" + split)[idx].to(self.device)
        processed_path = str(data.path)
        queried_layout = self._load_raw_layout(processed_path, complete_graph)
        assert solver is not None
        result_layout, score = solver.solve_with_trials(queried_layout, 3)
        self.writer.add_scalar("Score/" + split, score, self.epoch)
        # self.logger.info("score {} {} {}".format(split, score, self.epoch))
        result_layout.predict_probs = result_layout.predict
        assert plotter is not None
        img = result_layout.show_predict(plotter, None, True, True)
        self.writer.add_image(split + "/" + str(data.idx),
                              img,
                              self.epoch,
                              dataformats='HWC')

    def train(self,
              plotter: Plotter = None,
              solver: BaseSolver = None,
              complete_graph: TileGraph = None):
        self.logger.info("training start")
        while self.epoch < self.total_train_epoch:
            self.train_single_epoch(plotter, solver, complete_graph)
            # self.evaluate(plotter, complete_graph, solver, "train")
            # self.evaluate(plotter, complete_graph, solver, "test")
            self.epoch += 1
        self.logger.info("training done\n\n")
