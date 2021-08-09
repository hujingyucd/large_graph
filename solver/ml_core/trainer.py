import os
import sys
import logging
import torch
from torch_geometric.data import DataLoader
from solver.ml_core.losses import AreaLoss, OverlapLoss
from solver.ml_solver import MLSolver
from solver.MIS.greedy_solver import GreedySolver
from utils.graph_utils import sample_solution_greedy


class Trainer():
    def __init__(self,
                 network,
                 dataset_train,
                 dataset_test,
                 device,
                 model_save_path,
                 optimizer,
                 writer,
                 logger_name="trainer",
                 loss_weights=None,
                 resume=True,
                 total_train_epoch=100,
                 save_model_per_epoch=5):

        self.device = device
        self.model_save_path = model_save_path

        self.network = network
        self.optimizer = optimizer

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
        else:
            self.area_loss = AreaLoss()
            self.collision_loss = OverlapLoss()
        self.solution_loss = torch.nn.CrossEntropyLoss()

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

    def save(self):
        try:
            os.mkdir(self.model_save_path)
        except FileExistsError:
            pass
        torch.save(
            self.network.state_dict(),
            os.path.join(self.model_save_path,
                         "model_{}.pth".format(self.epoch)))
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(self.model_save_path,
                         "optimizer_{}.pth".format(self.epoch)))
        data_dict = {
            "epoch": self.epoch,
            "min_test_loss": self.min_test_loss,
            "min_test_loss_epoch": self.min_test_loss_epoch
        }

        torch.save(
            data_dict,
            os.path.join(self.model_save_path,
                         "datadict_{}.pth".format(self.epoch)))

        torch.save(data_dict, os.path.join(self.model_save_path, "latest.pth"))

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

    def greedy_based_solution(self, dataset, probs):
        print("ml solver:")
        ml_solver = MLSolver()
        ml_solver.eval(dataset, probs)
        print("\n\n")
        print("greedy solver:")
        greedy_solver = GreedySolver()
        greedy_solver.eval(dataset, probs)

        return

    def test_single_epoch(self, loader):
        # self.network.eval()
        area_losses = []
        collision_losses = []
        j = 0
        for batch in loader:
            torch.cuda.empty_cache()
            data = batch.to(self.device)
            probs = self.network(x=data.x, col_e_idx=data.edge_index)
            if self.epoch % 20 == 0 and self.epoch > 0 and j < 10:
                self.greedy_based_solution(batch, probs.cpu().detach().numpy())
                j = j + 1
            area_losses.append(self.area_loss(probs).detach())
            collision_losses.append(
                self.collision_loss(probs, data.edge_index).detach())
        area_losses = torch.stack(area_losses)
        collision_losses = torch.stack(collision_losses)
        losses = area_losses * collision_losses
        return torch.mean(losses), torch.mean(area_losses), torch.mean(
            collision_losses)

    def train_single_epoch(self):
        self.logger.info("training epoch start")
        torch.cuda.empty_cache()
        i = self.epoch
        self.network.train()
        for batch in self.loader_train:
            # get prediction
            data = batch.to(self.device)
            probs = self.network(x=data.x, col_e_idx=data.edge_index)
            try:
                assert not torch.any(torch.isnan(probs))
            except AssertionError:
                torch.save(probs, "./probs.pt")
                self.save()
                sys.exit("probs, data: {}".format(data.idx))

            loss_area = self.area_loss(probs)
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

            # solution = torch.where(probs > 0.5, 1.0, 0.0)
            # solution = torch.bernoulli(probs)
            with torch.no_grad():
                _, mask = sample_solution_greedy(data, probs)
                solution = torch.where(mask, 1.0, 0.0)
                score = self.area_loss(solution).detach(
                ) * self.collision_loss(solution, data.edge_index).detach()
                solution = solution.long()
            if score < train_loss:
                loss_solution = self.solution_loss(
                    torch.cat((1 - probs, probs), dim=1), solution)
            else:
                loss_solution = torch.tensor(0.0)
            train_loss += loss_solution

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            self.logger.debug(
                "{}, loss {:6f}, score {:6f}, loss_sol {:6f}".format(
                    i, train_loss.item(), score, loss_solution.item()))

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

    def train(self):
        self.logger.info("training start")
        while self.epoch < self.total_train_epoch:
            self.train_single_epoch()
            self.epoch += 1
        self.logger.info("training done\n\n")
