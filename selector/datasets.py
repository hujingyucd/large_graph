import logging
import torch
from torch_geometric.data import Dataset
# import numpy as np
import os
from utils.data_util import write_bricklayout
from utils.data_util import load_bricklayout
from tiling.brick_layout import BrickLayout
from tiling.tile_graph import TileGraph


class SampleGraphDataset(Dataset):
    def __init__(self,
                 root,
                 complete_graph: TileGraph,
                 device='cuda',
                 split="train",
                 subgraph_num=2000,
                 logger_name="DATASET",
                 transform=None,
                 pre_transform=None):
        self.logger = logging.getLogger(logger_name)
        self.split = split
        self.subgraph_num = subgraph_num
        self.complete_graph = complete_graph
        self.device = device
        super(SampleGraphDataset, self).__init__(root, transform,
                                                 pre_transform)

    @property
    def raw_file_names(self):
        return [
            os.path.join(self.split, f"data_{i}.pkl")
            for i in range(self.subgraph_num)
        ]

    @property
    def processed_file_names(self):
        return [
            os.path.join(self.split, 'data_{}.pkl'.format(i))
            for i in range(self.subgraph_num)
        ]

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        self.logger.error("please generate tile graph data first")
        raise FileNotFoundError("tile graph data not found")

    def process(self):
        self.logger.info("processing the data...")
        if not os.path.exists(os.path.join(self.processed_dir, self.split)):
            os.mkdir(os.path.join(self.processed_dir, self.split))
        for raw_path in self.raw_file_names:
            target_path = os.path.splitext(
                os.path.join(self.processed_dir, raw_path))[0] + ".pkl"
            if os.path.exists(target_path):
                continue
            try:
                layout = load_bricklayout(os.path.join(self.raw_dir, raw_path),
                                          None, self.device)
            except Exception as e:
                self.logger.error("read {} {}".format(str(e), raw_path))

            try:
                data = BrickLayout(
                    complete_graph=None,
                    node_feature=layout.node_feature[:, -1:],
                    collide_edge_index=layout.collide_edge_index,
                    collide_edge_features=layout.collide_edge_features,
                    align_edge_index=torch.tensor([[]]),
                    align_edge_features=torch.tensor([]),
                    re_index=layout.re_index,
                    target_polygon=layout.target_polygon)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                self.logger.debug("{}, node: {}, edge: {}".format(
                    target_path, len(data.node_feature),
                    data.collide_edge_index.size(1)))
                write_bricklayout(*(os.path.split(target_path)), data)
            except Exception as e:
                self.logger.error("{} {}".format(str(e), raw_path))

    def get(self, idx):
        data_path = os.path.join(self.processed_dir, self.split,
                                 'data_{}.pkl'.format(idx))
        try:
            data = load_bricklayout(data_path,
                                    complete_graph=self.complete_graph,
                                    device=self.device)
            data.idx = idx

        except FileNotFoundError:
            if idx != 0:
                self.logger.warning(
                    "data {} not found, using idx 0 instead".format(idx))
                data = self.get(0)
            else:
                raise FileNotFoundError("data 0 not found")
        return data
