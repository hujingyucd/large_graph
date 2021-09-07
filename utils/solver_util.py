import os
import numpy as np
import shapely
from shapely.geometry import Polygon
from collections import OrderedDict

from utils.algo_util import softmax
from utils.data_util import write_bricklayout
from utils.metrics import coverage_score

from tiling.brick_layout import BrickLayout

EPS = 1e-7


def exist_hole(origin_layout, origin_idx, new_predict):
    index_in_complete_graph = origin_layout.inverse_index[origin_idx]
    new_tile = origin_layout.complete_graph.tiles[
        index_in_complete_graph].tile_poly
    new_polygon = new_predict.current_polygon.union(new_tile.buffer(EPS))
    if isinstance(new_polygon, shapely.geometry.polygon.Polygon):
        if len(list(new_polygon.interiors)) > 0:
            return True
    elif isinstance(new_polygon, shapely.geometry.multipolygon.MultiPolygon):
        if any([
                len(list(new_polygon[i].interiors)) > 0
                for i in range(len(new_polygon))
        ]):
            return True
    else:
        print("error occurs in hole checking!!!!")
        input()

    return False


def get_nodes_order_array(prob_m, top_k):
    # sampling from the distribution if top k != 1
    # if top_k == 1:
    #     sampled_elem_array = np.argsort(-prob_m)
    # else:
    ALPHA = 100.0
    sampling_prob = np.clip(softmax(prob_m * ALPHA),
                            a_min=1e-7,
                            a_max=1 - 1e-7)
    sampling_prob = sampling_prob / sum(sampling_prob)
    assert len(sampling_prob) > 0
    sampled_elem_array = np.random.choice(len(sampling_prob),
                                          len(sampling_prob),
                                          replace=False,
                                          p=sampling_prob)
    return sampled_elem_array


def save_temp_layout(layout_cnt,
                     temp_layout,
                     tree_search_layout_dir,
                     plotter=None):
    write_bricklayout(tree_search_layout_dir, f'{layout_cnt}_data.pkl',
                      temp_layout)
    if plotter is not None:
        temp_layout.show_candidate_tiles(plotter=plotter,
                                         file_name=os.path.join(
                                             tree_search_layout_dir,
                                             f'{layout_cnt}_data.png'))


def label_collision_neighbor(collision_edges, new_predict, origin_idx,
                             origin_layout):
    # label neigbhour
    if not len(collision_edges) == 0:
        neighbor_collid_tiles = collision_edges[1][collision_edges[0] ==
                                                   origin_idx]
    else:
        neighbor_collid_tiles = []
    for adj_n in neighbor_collid_tiles:
        if adj_n in new_predict.unlabelled_nodes:
            assert adj_n not in new_predict.labelled_nodes
            new_predict.label_node(adj_n, 0, origin_layout)

    return new_predict


def create_solution(new_predict, origin_layout: BrickLayout, device):
    # calculate the score
    temp_sol = np.zeros(origin_layout.node_feature.shape[0])
    order_predict = []
    for key, value in new_predict.labelled_nodes.items():
        if value == 1:
            temp_sol[key] = 1
            order_predict.append(key)

    # evaluate the quality of a collision-free solution (origin)
    # score = Losses.solution_score(temp_sol, origin_layout)

    score = coverage_score(temp_sol, origin_layout, device=device)

    return score, temp_sol, order_predict


class SelectionSolution():
    def __init__(self, node_num):
        self.labelled_nodes = OrderedDict(sorted({}.items()))
        self.unlabelled_nodes = OrderedDict(
            sorted({key: 1.0
                    for key in range(node_num)}.items()))
        self.current_polygon = Polygon([])

    def label_node(self, node_idx, node_label, brick_layout):
        self.labelled_nodes[node_idx] = node_label
        self.unlabelled_nodes.pop(node_idx)
        # update polygon if the node_label is 1
        if node_label == 1:
            node_index_in_complete_graph = brick_layout.inverse_index[node_idx]
            self.current_polygon = self.current_polygon.union(
                brick_layout.complete_graph.
                tiles[node_index_in_complete_graph].tile_poly.buffer(EPS))
