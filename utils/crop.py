import torch
import numpy as np
from tiling.brick_layout import BrickLayout
from shapely.geometry import Polygon
from tiling.tile_factory import generatePolygon
from utils.algo_util import contain


def get_all_placement_in_polygon(graph: BrickLayout, polygon: Polygon):
    """
    Usually sub_nodes and tiles_super_set are equal, sometimes
    sub_nodes contains less nodes.
    """
    # get all tile placements in the supergraph
    tiles_super_set_mask = torch.zeros(len(graph.tiles), dtype=torch.bool)
    for i, t in enumerate(graph.tiles):
        if contain(polygon, t.tile_poly):
            tiles_super_set_mask[i] = True
    tiles_super_set = torch.arange(
        tiles_super_set_mask.size(0))[tiles_super_set_mask]

    sub_nodes_mask = torch.zeros(graph.node_feature.size(0), dtype=torch.bool)

    collide_edge_mask = torch.logical_and(
        tiles_super_set_mask[graph.collide_edge_index[0]],
        tiles_super_set_mask[graph.collide_edge_index[1]])
    sub_nodes_mask[graph.collide_edge_index[0][collide_edge_mask]] = True
    sub_nodes_mask[graph.collide_edge_index[1][collide_edge_mask]] = True

    sub_nodes = torch.arange(sub_nodes_mask.size(0))[sub_nodes_mask]
    filtered_collided_edges = graph.collide_edge_index[:, collide_edge_mask]

    if graph.align_edge_index.size(1) > 0:
        adj_edge_mask = torch.logical_and(
            tiles_super_set_mask[graph.align_edge_index[0]],
            tiles_super_set_mask[graph.align_edge_index[1]])
        filtered_adj_edges = graph.align_edge_index[:, adj_edge_mask]
    else:
        adj_edge_mask = torch.empty(0)
        filtered_adj_edges = torch.empty((1, 0))

    # assert len(sub_nodes) != 0

    return (tiles_super_set, filtered_collided_edges, filtered_adj_edges,
            collide_edge_mask, adj_edge_mask, sub_nodes)


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
                   radius: float = 4.2,
                   number_of_vertices: int = 30):

    # x_min, x_max, y_min, y_max = get_graph_bound(graph)
    # base_radius can be very small for thin shapes
    # base_radius = min(x_max - x_min, y_max - y_min) / 2
    irregularity = 0
    spikeyness = 0

    tile = graph.tiles[node]
    cords = np.array(tile.tile_poly.exterior.coords)
    node_x_min = np.min(cords[:, 0])
    node_x_max = np.max(cords[:, 0])
    node_y_min = np.min(cords[:, 1])
    node_y_max = np.max(cords[:, 1])

    while True:
        # generation of the random polygon
        # vertices = generatePolygon(
        #     (node_x_min + node_x_max) / 2, (node_y_min + node_y_max) / 2,
        #     max(base_radius * radius_random, node_y_max - node_y_min,
        #         node_x_max - node_x_min), irregularity, spikeyness,
        #     number_of_vertices)
        vertices = generatePolygon(
            (node_x_min + node_x_max) / 2, (node_y_min + node_y_max) / 2,
            radius, irregularity, spikeyness, number_of_vertices)
        polygon = Polygon(vertices)

        # or use get_all_placement_in_polygon()
        tiles_super_set = torch.tensor([
            i for i, t in enumerate(graph.tiles)
            if contain(polygon, t.tile_poly)
        ])

        if (tiles_super_set == node).any():
            break

    return tiles_super_set
