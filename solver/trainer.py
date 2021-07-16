import os
import sys
import torch
import numpy as np
from torch_geometric.data import DataLoader
import traceback
from torch.utils.tensorboard import SummaryWriter

from utils.datasets import GraphDataset
from solver.losses import Losses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer():
    def __init__(self, device, network, optimizer, data_path, model_save_path):
        self.device = device
        self.network = network
        self.optimizer = optimizer
        self.data_path = data_path
        self.model_save_path = model_save_path

        self.writer = SummaryWriter(log_dir="logs")

        self.epoch = 0
        self.total_train_epoch = 10000
        self.min_test_loss = float("inf")
        self.min_test_loss_epoch = 0

        self.load()

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
        

    def train(self,
              optimizer,
              batch_size=32,
              training_epoch=10000,
              save_model_per_epoch=5):

        # print(self.data_path)
        dataset_train = GraphDataset(root=self.data_path, split="train", subgraph_num=200)
        dataset_test = GraphDataset(root=self.data_path, split="test", subgraph_num=30)
        loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
        loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

        self.total_train_epoch = training_epoch

        print("Training Start!!!", flush=True)
        while self.epoch < self.total_train_epoch:
            torch.cuda.empty_cache()
            i = self.epoch
            self.network.train()
            for batch in loader_train:
                ## get prediction
                data = batch.to(self.device)
                probs = self.network(x = data.x,
                                    col_e_idx = data.edge_index)

                # optimizer.zero_grad()
                train_loss, *_ = Losses.calculate_unsupervised_loss(probs, data.x, data.edge_index)
                optimizer.zero_grad()                                                                
                train_loss.backward()
                optimizer.step()


            # self.network.train()
            torch.cuda.empty_cache()
            loss_train, loss_train_area, loss_train_coll = Losses.cal_avg_loss(self.network, loader_train)
            self.writer.add_scalar("Loss/train", loss_train, i)
            self.writer.add_scalar("AreaLoss/train", loss_train_area, i)
            self.writer.add_scalar("CollisionLoss/train", loss_train_coll, i)
            print(f"epoch {i}: training loss: {loss_train}", flush=True)
            
            loss_test, loss_test_area, loss_test_coll  = Losses.cal_avg_loss(self.network, loader_test)
            self.writer.add_scalar("Loss/test", loss_test, i)
            self.writer.add_scalar("AreaLoss/test", loss_test_area, i)
            self.writer.add_scalar("CollisionLoss/test", loss_test_coll, i)
            print(f"epoch {i}: testing loss: {loss_test}", flush=True)

            ############# result debugging #############
            if (loss_test < self.min_test_loss or i % save_model_per_epoch == 0):
                if loss_test < self.min_test_loss:
                    self.min_test_loss = loss_test
                    self.min_test_loss_epoch = i
                torch.cuda.empty_cache()
                ############# network testing #############
                # self.network.train()

                ############# model save #############
                torch.save(self.network.state_dict(), os.path.join(self.model_save_path, f'model_{i}_{loss_test}.pth'))
                torch.save(optimizer.state_dict(), os.path.join(self.model_save_path, f'optimizer_{i}_{loss_test}.pth'))
                data_dict = {
                    "epoch": i,
                    "min_test_loss": self.min_test_loss,
                    "min_test_loss_epoch": self.min_test_loss_epoch
                }
                torch.save(data_dict, os.path.join(self.model_save_path, "latest.pth"))
                print(f"model saved at epoch {i}")

            self.epoch +=1


        print("Training Done!!!")
