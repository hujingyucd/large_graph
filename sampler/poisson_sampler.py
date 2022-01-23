from typing import Tuple
import torch
import numpy as np
from shapely.geometry import Point
from sampler.base_sampler import Sampler
from torch_geometric.utils import subgraph
from tiling.brick_layout import BrickLayout
import random


def get_graph_bound(graph: BrickLayout):
    tiles = [np.array(t.tile_poly.exterior.coords) for t in graph.tiles]

    # getting the bound
    x_min = np.min([np.min(tile[:, 0]) for tile in tiles])
    x_max = np.max([np.max(tile[:, 0]) for tile in tiles])
    y_min = np.min([np.min(tile[:, 1]) for tile in tiles])
    y_max = np.max([np.max(tile[:, 1]) for tile in tiles])
    return x_min, x_max, y_min, y_max


class PoissonSampler(Sampler):

    def __init__(self, radius: float = 2.1):
        super(PoissonSampler, self).__init__()
        # sample radius
        self.radius = radius

    def centroid(self, vertexes):
        x_list = [vertex[0] for vertex in vertexes]
        y_list = [vertex[1] for vertex in vertexes]
        x = sum(x_list) / len(vertexes)
        y = sum(y_list) / len(vertexes)
        return (x, y)

    def cal_distance(self, sample, locs):
        x, y = self.centroid(locs)
        d = ((sample[0] - x)**2 + (sample[1] - y)**2)**0.5
        return d

    def get_neighbors(self, coords):
        options = [(0, 0), (-1, -2), (0, -2), (1, -2), (-2, -1), (-1, -1),
                   (0, -1), (1, -1), (2, -1), (-2, 0), (-1, 0), (1, 0), (2, 0),
                   (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1), (-1, 2), (0, 2),
                   (1, 2)]
        neighbors = []
        for delta_x, delta_y in options:
            neig_coords = coords[0] + delta_x, coords[1] + delta_y
            if (neig_coords[0] < 0 or neig_coords[0] >= self.rows
                    or neig_coords[1] < 0 or neig_coords[1] >= self.cols):
                continue
            neig_cell = self.cells[neig_coords]
            if neig_cell != -1:
                neighbors.append(neig_cell)
        return neighbors

    def check_valid(self, pt):
        pt_obj = Point(pt[0], pt[1])

        if not pt_obj.within(self.poly):
            return False

        coords = self.get_cell_coords(pt)
        for index in self.get_neighbors(coords):
            target_pt = self.samples[index]
            distance = (target_pt[0] - pt[0])**2 + (target_pt[1] - pt[1])**2
            if distance < self.radius**2:
                return False
        return True

    def get_new_pt(self, pt):
        trial = 0
        while True:
            r = random.uniform(self.radius, 2 * self.radius)
            theta = random.uniform(0, 2 * np.pi)
            new_pt = (pt[0] + r * np.sin(theta), pt[1] + r * np.cos(theta))

            if self.check_valid(new_pt):
                return new_pt

            trial += 1
            if trial >= 30:
                break
        return False

    def get_cell_coords(self, pt):
        row_idx = int((pt[0] - self.x_min) // self.edge)
        col_idx = int((pt[1] - self.y_min) // self.edge)
        return row_idx, col_idx

    def sample(self, graph: BrickLayout) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        graph: original full graph
        return:
            new_edges
            node_mask
        """
        graph.update_tiles()
        self.samples = []
        self.poly = graph.show_super_contour(None, None)

        self.x_min, self.x_max, self.y_min, self.y_max = get_graph_bound(graph)
        self.width = self.x_max - self.x_min
        self.height = self.y_max - self.y_min

        self.edge = self.radius / (2**0.5)
        self.rows = int(self.width / self.edge) + 1
        self.cols = int(self.height / self.edge) + 1

        coords_list = [(ix, iy) for ix in range(self.rows)
                       for iy in range(self.cols)]
        self.cells = {coords: -1 for coords in coords_list}

        first_pt = (random.uniform(self.x_min, self.x_max),
                    random.uniform(self.y_min, self.y_max))
        pt_obj = Point(first_pt[0], first_pt[1])
        while not pt_obj.within(self.poly):
            first_pt = (random.uniform(self.x_min, self.x_max),
                        random.uniform(self.y_min, self.y_max))
            pt_obj = Point(first_pt[0], first_pt[1])

        self.samples.append(first_pt)
        self.cells[self.get_cell_coords(first_pt)] = 0
        active_list = [0]

        # sample new points
        while active_list:
            idx = random.choice(active_list)
            pt = self.samples[idx]
            new_pt = self.get_new_pt(pt)

            if new_pt:
                self.samples.append(new_pt)
                active_list.append(len(self.samples) - 1)
                self.cells[self.get_cell_coords(new_pt)] = len(
                    self.samples) - 1
            else:
                active_list.remove(idx)
        # print(f'{len(self.samples)} points sampled.')

        # map sampled coords to tiles
        nodes = []
        for i in range(len(self.samples)):
            point = self.samples[i]
            point_obj = Point(point[0], point[1])
            d_min = ((self.x_max - self.x_min)**2 +
                     (self.y_max - self.y_min)**2)**0.5
            tile_min = None
            for j in range(len(graph.tiles)):
                t = graph.tiles[j]
                if not point_obj.within(t.tile_poly):
                    continue

                locs = list(t.tile_poly.exterior.coords)
                d = self.cal_distance(point, locs[:-1])
                if d < d_min:
                    d_min = d
                    tile_min = j
            if tile_min is not None:
                nodes.append(tile_min)

        final_nodes = torch.unique(torch.tensor(nodes, dtype=torch.long),
                                   sorted=True)
        # assert len(self.samples) == len(nodes)
        new_edges, _ = subgraph(final_nodes,
                                graph.collide_edge_index,
                                num_nodes=graph.node_feature.size(0),
                                relabel_nodes=True)

        node_mask = torch.zeros(graph.node_feature.size(0), dtype=torch.bool)
        node_mask[final_nodes] = True
        return new_edges, node_mask
