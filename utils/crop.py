import torch
import numpy as np
import random
from collections import defaultdict

from tiling.brick_layout import BrickLayout
from shapely.geometry import Polygon
from tiling.tile_factory import generatePolygon
from utils.algo_util import contain


def generate_brick_layout_data(graph: BrickLayout, super_tiles: list,
                               collide_edges, adj_edges, collide_edge_mask,
                               adj_edge_mask):
    # edge feature
    super_edge_features_collide = graph.collide_edge_features[
        collide_edge_mask]
    super_edge_features_adj = graph.align_edge_features[adj_edge_mask]

    for edge in super_edge_features_collide:
        assert edge is not None

    # re-index: mapping from complete graph to the current super graph
    re_index = defaultdict(int)
    for i in range(len(super_tiles)):
        re_index[super_tiles[i]] = i

    # node feature
    # node_feature = torch.zeros((len(super_tiles), graph.node_feature.size(1)))
    # for i, tile_idx in enumerate(super_tiles):
    #     current_tile = graph.tiles[tile_idx]
    #     # node_feature[i][current_tile.id] = 1
    #     node_feature[i][-1] = current_tile.area() / max(
    #         [t.area() for t in graph.tiles])

    return (None, collide_edges, super_edge_features_collide, adj_edges,
            super_edge_features_adj, re_index)


def get_all_placement_in_polygon(graph: BrickLayout, polygon: Polygon):
    # get all tile placements in the supergraph
    tiles_super_set = [
        i for i in range(len(graph.tiles))
        if contain(polygon, graph.tiles[i].tile_poly)
    ]

    collide_edge = torch.transpose(graph.collide_edge_index, 0, 1)
    adj_edge = torch.transpose(graph.align_edge_index, 0, 1)

    sub_nodes = []
    collide_edge_mask = []
    for i in range(len(collide_edge)):
        edge = collide_edge[i]
        if edge[0].item() in tiles_super_set and edge[1].item(
        ) in tiles_super_set:
            collide_edge_mask.append(True)
            if edge[0].item() not in sub_nodes:
                sub_nodes.append(edge[0].item())
            if edge[1].item() not in sub_nodes:
                sub_nodes.append(edge[1].item())
        else:
            collide_edge_mask.append(False)

    adj_edge_mask = []
    for i in range(len(adj_edge)):
        edge = adj_edge[i]
        if edge[0].item() in tiles_super_set and edge[1].item(
        ) in tiles_super_set:
            adj_edge_mask.append(True)
        else:
            adj_edge_mask.append(False)

    filtered_collided_edges = collide_edge[collide_edge_mask]
    filtered_adj_edges = adj_edge[adj_edge_mask]
    filtered_collided_edges = torch.transpose(filtered_collided_edges, 0, 1)
    filtered_adj_edges = torch.transpose(filtered_adj_edges, 0, 1)

    # assert len(sub_nodes) != 0

    return (tiles_super_set, filtered_collided_edges, filtered_adj_edges,
            collide_edge_mask, adj_edge_mask, torch.tensor(sub_nodes))


def create_brick_layout_from_polygon(graph: BrickLayout, polygon: Polygon):

    # get filter graph
    (tiles_super_set, filtered_collided_edges, filtered_adj_edges,
     collide_edge_mask, adj_edge_mask,
     sub_nodes) = get_all_placement_in_polygon(graph, polygon)

    # produce data needed
    (node_feature, collide_edge_index, collide_edge_features, align_edge_index,
     align_edge_features, re_index) = generate_brick_layout_data(
         graph, tiles_super_set, filtered_collided_edges, filtered_adj_edges,
         collide_edge_mask, adj_edge_mask)

    return (node_feature, collide_edge_index, collide_edge_features,
            align_edge_index, align_edge_features, re_index, sub_nodes)


def get_graph_bound(graph: BrickLayout):
    tiles = [np.array(t.tile_poly.exterior.coords) for t in graph.tiles]

    # getting the bound
    x_min = np.min([np.min(tile[:, 0]) for tile in tiles])
    x_max = np.max([np.max(tile[:, 0]) for tile in tiles])
    y_min = np.min([np.min(tile[:, 1]) for tile in tiles])
    y_max = np.max([np.max(tile[:, 1]) for tile in tiles])
    return x_min, x_max, y_min, y_max


def crop_2d_circle(node,
                   graph: BrickLayout,
                   max_vertices: float = 10,
                   low=0.2,
                   high=0.7,
                   plotter=None,
                   plot_shape=False):

    # try until can create
    count = 0
    while True:
        count += 1
        if count > 10:
            return None, None, None, None, None, None, None
        x_min, x_max, y_min, y_max = get_graph_bound(graph)
        base_radius = min(x_max - x_min, y_max - y_min) / 2
        radius_random = random.uniform(low, high)
        # irregularity = random.random()
        # spikeyness = random.random()
        irregularity = 0
        spikeyness = 0
        number_of_vertices = random.randint(3, max_vertices)

        tile = graph.tiles[node]
        cords = np.array(tile.tile_poly.exterior.coords)
        node_x_min = np.min(cords[:, 0])
        node_x_max = np.max(cords[:, 0])
        node_y_min = np.min(cords[:, 1])
        node_y_max = np.max(cords[:, 1])

        # generation of the random polygon
        vertices = generatePolygon(
            (node_x_min + node_x_max) / 2, (node_y_min + node_y_max) / 2,
            base_radius * radius_random, irregularity, spikeyness,
            number_of_vertices)
        polygon = Polygon(vertices)

        if plot_shape:
            assert plotter is not None
            plotter.draw_polys('./polys/generated_shape.png',
                               [('green', np.array(vertices))])

        (node_feature, collide_edge_index, collide_edge_features,
         align_edge_index, align_edge_features, re_index,
         sub_nodes) = create_brick_layout_from_polygon(graph, polygon)

        # skip
        if len(collide_edge_index) == 0:
            print("skip")
            continue

        return (node_feature, collide_edge_index, collide_edge_features,
                align_edge_index, align_edge_features, re_index, sub_nodes)
