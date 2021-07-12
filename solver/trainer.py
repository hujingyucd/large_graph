import os
import torch
import numpy as np
from utils.datasets import GraphDataset
from torch_geometric.data import DataLoader
import time
import glob
import asyncio, concurrent.futures
import multiprocessing as mp
import inputs.config as config
import traceback
from solver.losses import Losses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer():
    def __init__(self, debugger, plotter, device, network, data_path):
        self.debugger = debugger
        self.plotter = plotter
        self.device = device
        self.model_save_path = self.debugger.file_path(self.debugger.file_path("model"))
        self.data_path = data_path
        self.training_path = os.path.join(data_path, 'train')
        self.testing_path = os.path.join(data_path, 'test')
        self.network = network

        # creation of directory for result
        os.mkdir(self.debugger.file_path('model'))
        os.mkdir(self.debugger.file_path('result'))


    def train(self,
              ml_solver,
              optimizer,
              batch_size=32,
              training_epoch=10000,
              save_model_per_epoch=5):

        dataset_train = GraphDataset(root=self.training_path)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dataset_test = GraphDataset(root=self.testing_path)
        loader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)

        print("Training Start!!!", flush=True)
        for i in range(training_epoch):
            self.network.train()
            for batch in loader_train:
                ## get prediction
                data = batch.to(self.device)
                probs = self.network(x = data.x,
                                    col_e_idx = data.edge_index,
                                    col_e_features = data.edge_features)
                try:
                    optimizer.zero_grad()
                    train_loss, *_ = Losses.calculate_unsupervised_loss(probs, data.x, data.collide_edge_index,
                                                                                    adj_edges_index=data.edge_index,
                                                                                    adj_edge_features=data.edge_features)
                    train_loss.backward()
                    optimizer.step()
                except:
                    print(traceback.format_exc())
                    continue

            # self.network.train()
            torch.cuda.empty_cache()
            loss_train, *_ = Losses.cal_avg_loss(self.network, loader_train)
            print(f"epoch {i}: training loss: {loss_train}", flush=True)
            loss_test, *_  = Losses.cal_avg_loss(self.network, loader_test)
            print(f"epoch {i}: testing loss: {loss_test}", flush=True)


        print("Training Done!!!")
