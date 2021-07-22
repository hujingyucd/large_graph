import logging
from random import randint
import torch
from torch_geometric.data import Dataset, Data, download_url
from torch_geometric.utils import subgraph
# import numpy as np
import os
import gzip


def gen_subgraph(graph, size_min, size_max, node_idx, expected_size):
    current_nodes = set([node_idx])
    current_size = len(current_nodes)

    row, col = graph["edge_index"]
    node_mask = row.new_empty(graph.num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    node_idx = torch.tensor([node_idx], device=row.device)
    subsets = [node_idx]

    # slow BFS
    while current_size < expected_size:
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        new_nodes = col[edge_mask]
        subsets.append(new_nodes)
        prev_size = current_size
        current_nodes.update(set(new_nodes.tolist()))
        current_size = len(current_nodes)
        if current_size == prev_size:
            break

    subset = torch.cat(subsets).unique()
    while current_size > size_max:
        subsets = subsets[:-1]
        subset = torch.cat(subsets).unique()
        current_size = len(subset)

    if current_size < size_min:
        return None

    # relabel nodes
    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]
    edge_index = graph["edge_index"][:, edge_mask]
    node_idx = row.new_full((graph.num_nodes, ), -1)
    node_idx[subset] = torch.arange(subset.size(0), device=row.device)
    edge_index = node_idx[edge_index]

    data = Data(torch.ones(len(subset), 1), edge_index=edge_index)
    data.num_nodes = len(subset)
    return data


class GraphDataset(Dataset):
    def __init__(self,
                 root,
                 url=None,
                 split="train",
                 subgraph_num=200,
                 subgraph_size_min=1000,
                 subgraph_size_max=30000,
                 transform=None,
                 pre_transform=None):
        self.subgraph_num = subgraph_num
        self.subgraph_size_min = subgraph_size_min
        self.subgraph_size_max = subgraph_size_max
        self.url = url
        self.split = split
        super(GraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        files_list = ["complete_graph.pt"]
        return files_list

    @property
    def processed_file_names(self):
        return [
            os.path.join(self.split, 'data_{}.pt'.format(i))
            for i in range(self.subgraph_num)
        ]

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        raw_data_path = os.path.join(self.raw_dir, "raw_edges.txt.gz")
        if not os.path.exists(raw_data_path):
            try:
                os.remove(self.raw_dir)
            except OSError:
                pass
            download_url(self.url, self.raw_dir)
            saved_dir = os.getcwd()
            os.chdir(self.raw_dir)
            os.rename(next(os.walk('./'))[2][0], "raw_edges.txt.gz")
            os.chdir(saved_dir)
        logging.info("downloaded")

        edge_index = []
        with gzip.open(raw_data_path, 'rb') as f:
            for i, line in enumerate(f):
                line = line.decode().strip().split()
                try:
                    edge_index.append([int(line[0]), int(line[1])])
                except ValueError:
                    pass
                if (i + 1) % 1000000 == 0:
                    logging.info("reading raw data, {}".format(i + 1))
        logging.info("read")
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_index.transpose_(0, 1)
        nodes = torch.unique(edge_index)
        logging.info("unique")
        # print(len(nodes))
        edge_index, _ = subgraph(nodes, edge_index, relabel_nodes=True)
        # print(len(edge_index))

        data = Data(edge_index=edge_index)
        data.num_nodes = data.num_nodes
        torch.save(data, os.path.join(self.raw_dir, 'complete_graph.pt'))

    def process(self):
        logging.info("processing the data...")
        if not os.path.exists(os.path.join(self.processed_dir, self.split)):
            os.mkdir(os.path.join(self.processed_dir, self.split))
        graph = torch.load(os.path.join(self.raw_dir, "complete_graph.pt"))
        graph.num_nodes = graph.num_nodes
        degree = graph.num_edges / graph.num_nodes
        assert degree > 2

        i = 0
        centers = set([])
        while i < len(self.processed_file_names):
            target_path = os.path.join(self.processed_dir, self.split,
                                       "data_{}.pt".format(i))
            if os.path.exists(target_path):
                logging.info("{}/data_{}.pt exists".format(self.split, i))
                i += 1
                continue
            node_idx = randint(0, graph.num_nodes - 1)
            while node_idx in centers:
                node_idx = randint(0, graph.num_nodes - 1)
            centers.add(node_idx)
            expected_size = randint(self.subgraph_size_min,
                                    self.subgraph_size_max)
            data = gen_subgraph(graph, self.subgraph_size_min,
                                self.subgraph_size_max, node_idx,
                                expected_size)
            if data is None:
                continue

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            logging.info(
                "subgraph {}, expected size: {}, node: {}, edge: {}".format(
                    i, expected_size, data.num_nodes, data.num_edges))

            torch.save(data, target_path)
            i += 1

    def get(self, idx):
        data = torch.load(
            os.path.join(self.processed_dir, self.split,
                         'data_{}.pt'.format(idx)))
        return data
