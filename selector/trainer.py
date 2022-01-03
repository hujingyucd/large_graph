import os
import logging
import random
import torch
from torch.utils.tensorboard import SummaryWriter

from selector.selection_solve import solve_by_sample_selection
from selector.crop_solve import solve_by_crop
from solver.MIS.base_solver import BaseSolver
from interfaces.qt_plot import Plotter
from sampler.base_sampler import Sampler


class SelectorTrainer():
    def __init__(self,
                 network,
                 dataset_train,
                 dataset_test,
                 device,
                 model_save_path,
                 optimizer,
                 sampler: Sampler,
                 solver: BaseSolver,
                 writer: SummaryWriter,
                 logger_name="trainer",
                 resume=True,
                 total_train_epoch=100,
                 save_model_per_epoch=5):

        self.device = device
        self.model_save_path = model_save_path

        self.network = network
        self.optimizer = optimizer

        self.solver = solver
        self.sampler = sampler

        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.loader_train = (self.dataset_train[i] for i in random.sample(
            range(len(self.dataset_train)), len(self.dataset_train)))
        self.loader_test = (self.dataset_train[i] for i in random.sample(
            range(len(self.dataset_train)), len(self.dataset_train)))

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

    def train_single_epoch(self, plotter: Plotter = None):
        i = self.epoch
        self.logger.info("training epoch {} start".format(i))
        torch.cuda.empty_cache()
        train_metrics = {"holes": [], "loss": []}
        self.network.train()
        for idx, data in enumerate(self.loader_train):
            log_items = [str(i), str(idx), "data {}".format(data.idx)]
            self.optimizer.zero_grad()
            self.network.train()

            epoch_path = os.path.join(os.path.split(self.logger.handlers[0].baseFilename)[0], 'epoch{}'.format(i))
            if not os.path.exists(epoch_path):
                os.mkdir(epoch_path)
            save_path = os.path.join(os.path.join(os.path.split(self.logger.handlers[0].baseFilename)[0], 'epoch{}'.format(i)),'data{}'.format(data.idx))
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            # final_layout, prob_records, solutions = solve_by_sample_selection(
            #     data,
            #     self.network,
            #     self.solver,
            #     self.sampler,
            #     show_intermediate=True)

            final_layout, prob_records, solutions = solve_by_crop(
                data,
                self.network,
                self.solver,
                self.sampler,
                show_intermediate=True,
                log_dir=save_path)

            # compute reward
            num_holes = final_layout.detect_holes()
            log_items.append("holes: {}".format(num_holes))
            train_metrics["holes"].append(num_holes)
            reward = -1 - num_holes

            # compute loss and optimize
            loss_reward = -reward * prob_records.log().sum()
            log_items.append("loss: {}".format(loss_reward.item()))
            train_metrics["loss"].append(loss_reward.item())

            # debugging
            if data.idx < 3:
                for step, sol in enumerate(solutions):
                    final_layout.predict = sol
                    _ = final_layout.show_predict(
                        plotter=plotter,
                        file_name=os.path.join(
                            os.path.split(
                                self.logger.handlers[0].baseFilename)[0],
                            "epoch{}_data{}_step{}.png".format(
                                i, data.idx, step)),
                        do_show_super_contour=True,
                        do_show_tiling_region=True)

            loss_reward.backward()
            self.optimizer.step()

            self.logger.debug(", ".join(log_items))

        for key, vs in train_metrics.items():
            self.writer.add_scalar("train/{}".format(key),
                                   sum(vs) / len(vs), self.epoch)

        self.logger.info("training epoch {} done\n".format(i))

        torch.cuda.empty_cache()

        if (i % self.save_model_per_epoch == 0):
            # if loss_test < self.min_test_loss:
            #     self.min_test_loss = loss_test
            #     self.min_test_loss_epoch = self.epoch
            self.save()
            self.logger.info("model saved at epoch {}".format(i))

    def train(self, plotter: Plotter = None):
        self.logger.info("training start")
        while self.epoch < self.total_train_epoch:
            self.loader_train = (self.dataset_train[i] for i in random.sample(
                range(len(self.dataset_train)), len(self.dataset_train)))
            self.loader_test = (self.dataset_train[i] for i in random.sample(
                range(len(self.dataset_train)), len(self.dataset_train)))
            try:
                self.train_single_epoch(plotter)
            except Exception as e:
                print("epoch: ", self.epoch)
                raise e
            self.epoch += 1
        self.logger.info("training done\n")
