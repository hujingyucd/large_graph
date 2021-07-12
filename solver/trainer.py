import os
import torch
import numpy as np
from utils.datasets import GraphDataset
from torch_geometric.data import DataLoader
import traceback
from solver.losses import Losses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer():
    def __init__(self, device, network, optimizer, data_path, model_save_path):
        self.device = device
        self.network = network
        self.optimizer = optimizer
        self.model_save_path = model_save_path,
        self.data_path = data_path
        self.training_path = os.path.join(data_path, 'train')
        self.testing_path = os.path.join(data_path, 'test')
        

    def train(self,
              ml_solver,
              optimizer,
              batch_size=32,
              training_epoch=10000,
              save_model_per_epoch=5):

        dataset_train = GraphDataset(root=self.data_path, split="train", subgraph_num=200)
        dataset_test = GraphDataset(root=self.data_path, split="test", subgraph_num=30)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

        print("Training Start!!!", flush=True)
        min_test_loss = float("inf")
        for i in range(training_epoch):
            self.network.train()
            for batch in loader_train:
                ## get prediction
                data = batch.to(self.device)
                probs = self.network(x = data.x,
                                    col_e_idx = data.edge_index)
                try:
                    optimizer.zero_grad()
                    train_loss, *_ = Losses.calculate_unsupervised_loss(probs, data.x, adj_edges_index=data.edge_index)
                                                                                    
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

            ############# result debugging #############
            if (loss_test < min_test_loss or i % save_model_per_epoch == 0):
                if loss_test < min_test_loss:
                    min_test_loss = loss_test
                torch.cuda.empty_cache()
                ############# network testing #############
                # self.network.train()


                torch.save(self.network.state_dict(), os.path.join(self.model_save_path, f'model_{i}_{loss_test}.pth'))
                torch.save(optimizer.state_dict(), os.path.join(self.model_save_path, f'optimizer_{i}_{loss_test}.pth'))
                print(f"model saved at epoch {i}")


        print("Training Done!!!")
