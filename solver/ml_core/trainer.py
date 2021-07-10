import os
import logging
import torch
from solver.ml_core.datasets import GraphDataset
from torch_geometric.data import DataLoader
from solver.ml_core.losses import AreaLoss, OverlapLoss
from torch.utils.tensorboard import SummaryWriter


class Trainer():
    def __init__(self,
                 network,
                 data_path,
                 device,
                 model_save_path,
                 optimizer,
                 total_train_epoch=10000,
                 save_model_per_epoch=5):

        self.device = device
        self.model_save_path = model_save_path
        self.data_path = data_path

        self.network = network
        self.optimizer = optimizer
        dataset_train = GraphDataset(root=data_path,
                                     split="train",
                                     subgraph_num=200)
        dataset_test = GraphDataset(root=data_path,
                                    split="test",
                                    subgraph_num=30)
        self.loader_train = DataLoader(dataset_train,
                                       batch_size=1,
                                       shuffle=True)
        self.loader_test = DataLoader(dataset_test,
                                      batch_size=1,
                                      shuffle=False)

        self.area_loss = AreaLoss(weight=1.0)
        self.collision_loss = OverlapLoss(weight=10.0)

        self.total_train_epoch = total_train_epoch
        self.writer = SummaryWriter(log_dir="logs")

        self.epoch = 0
        self.min_test_loss = float("inf")
        self.min_test_loss_epoch = 0

        self.load()

    def save(self):
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
        self.network.load_state_dict(
            os.path.join(self.model_save_path,
                         "model_{}.pth".format(self.epoch)))
        self.optimizer.load_state_dict(
            os.path.join(self.model_save_path,
                         "optimizer_{}.pth".format(self.epoch)))

    def test_single_epoch(self, loader):
        self.network.eval()
        area_losses = []
        collision_losses = []
        for batch in loader:
            data = batch.to(self.device)
            probs = self.network(x=data.x, col_e_idx=data.edge_index)
            area_losses.append(self.area_loss(probs))
            collision_losses.append(self.collision_loss(
                probs, data.edge_index))
        area_losses = torch.stack(area_losses)
        collision_losses = torch.stack(collision_losses)
        losses = area_losses * collision_losses
        return torch.mean(losses), torch.mean(area_losses), torch.mean(
            collision_losses)

    def train_single_epoch(self):
        torch.cuda.empty_cache()
        i = self.epoch
        self.network.train()
        for batch in self.loader_train:
            # get prediction
            data = batch.to(self.device)
            probs = self.network(x=data.x, col_e_idx=data.edge_index)

            train_loss = self.area_loss(probs) * self.collision_loss(
                probs, data.edge_index)
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

        torch.cuda.empty_cache()
        loss_train, loss_train_area, loss_train_coll = self.test_single_epoch(
            self.loader_train)
        self.writer.add_scalar("Loss/train", loss_train, self.epoch)
        self.writer.add_scalar("AreaLoss/train", loss_train_area, self.epoch)
        self.writer.add_scalar("CollisionLoss/train", loss_train_coll,
                               self.epoch)

        torch.cuda.empty_cache()
        loss_test, loss_test_area, loss_test_coll = self.test_single_epoch(
            self.loader_test)
        self.writer.add_scalar("Loss/test", loss_test, self.epoch)
        self.writer.add_scalar("AreaLoss/test", loss_test_area, self.epoch)
        self.writer.add_scalar("CollisionLoss/test", loss_test_coll,
                               self.epoch)

        logging.info("epoch {} testing loss {:.6f}".format(i, loss_test))

        # result debugging
        if (loss_test < self.min_test_loss
                or i % self.save_model_per_epoch == 0):
            if loss_test < self.min_test_loss:
                self.min_test_loss = loss_test
                self.min_test_loss_epoch = self.epoch
            torch.cuda.empty_cache()
            self.save()
            logging.info("model saved at epoch {}".format(i))

    def train(self):
        logging.info("training start")
        while self.epoch < self.total_train_epoch:
            self.train_single_epoch()
            self.epoch += 1
        logging.info("training done")
