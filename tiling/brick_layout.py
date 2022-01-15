from typing import List, Union, Dict, DefaultDict
from functools import reduce
import numpy as np
import torch
from shapely.ops import unary_union
import shapely
from collections import defaultdict
import copy
import networkx as nx
import matplotlib.pyplot as plt

from interfaces.qt_plot import Plotter
from tiling.tile_graph import TileGraph
from utils.algo_util import interp
# episolon for area err
EPS = 1e-5
BUFFER_TILE_EPS = EPS * 1e-5
SIMPLIFIED_TILE_EPS = BUFFER_TILE_EPS * 1e3


class BrickLayout():

    def __init__(self,
                 complete_graph: TileGraph,
                 node_feature: torch.Tensor,
                 collide_edge_index: torch.Tensor,
                 collide_edge_features: torch.Tensor,
                 align_edge_index: torch.Tensor,
                 align_edge_features: torch.Tensor,
                 re_index: Dict = {},
                 inverse_index: DefaultDict = defaultdict(int),
                 target_polygon=None):
        self.complete_graph = complete_graph
        self.node_feature = node_feature.float()
        self.collide_edge_index = collide_edge_index
        self.collide_edge_features = collide_edge_features.float()
        self.align_edge_index = align_edge_index
        self.align_edge_features = align_edge_features.float()
        self.tiles = None

        # assertion for brick_layout
        # align_edge_index_list = align_edge_index.T.tolist()
        # collide_edge_index_list = collide_edge_index.T.tolist()

        # assertion
        # for f in align_edge_index_list:
        #     assert [f[1], f[0]] in align_edge_index_list
        # for f in collide_edge_index_list:
        #     assert [f[1], f[0]] in collide_edge_index_list

        # mapping from index of complete graph to index of super graph
        if re_index:
            self.re_index = re_index
            # mapping from index of super graph to index of complete graph
            # self.inverse_index = defaultdict(int)
            self.inverse_index = {}
            for k, v in self.re_index.items():
                self.inverse_index[v] = k
        else:
            self.inverse_index = inverse_index
            self.re_index = {v: k for k, v in inverse_index.items()}

        self.predict = np.zeros(len(self.node_feature))
        self.predict_probs: Union[List[float], np.ndarray] = []
        self.predict_order: List[int] = []
        self.target_polygon = target_polygon

        # save super poly
        self.super_contour_poly = None

    def __deepcopy__(self, memo):
        new_inst = type(self).__new__(self.__class__)  # skips calling __init__
        new_inst.complete_graph = self.complete_graph
        new_inst.node_feature = self.node_feature
        new_inst.collide_edge_index = self.collide_edge_index
        new_inst.collide_edge_features = self.collide_edge_features
        new_inst.align_edge_index = self.align_edge_index
        new_inst.align_edge_features = self.align_edge_features
        new_inst.re_index = self.re_index
        new_inst.inverse_index = self.inverse_index
        new_inst.predict = copy.deepcopy(self.predict)
        new_inst.predict_probs = copy.deepcopy(self.predict_probs)
        new_inst.super_contour_poly = self.super_contour_poly
        new_inst.predict_order = self.predict_order
        new_inst.target_polygon = self.target_polygon

        return new_inst

    def is_solved(self):
        return len(self.predict) != 0

    # ALL PLOTTING FUNCTIONS #################
    def show_candidate_tiles(self,
                             plotter: Plotter,
                             file_name,
                             style="blue_trans"):
        tiles = self.complete_graph.tiles
        assert tiles is not None
        selected_indices = [k for k in self.re_index.keys()]
        selected_tiles = [tiles[s] for s in selected_indices]
        plotter.draw_contours(
            [tile.get_plot_attribute(style) for tile in selected_tiles],
            file_path=file_name)

    def show_predict(self,
                     plotter: Plotter,
                     file_name: str = None,
                     do_show_super_contour: bool = True,
                     do_show_tiling_region: bool = True):
        tiles = self.predict

        # show input polygon
        (tiling_region_exteriors,
         tiling_region_interiors) = BrickLayout.get_polygon_plot_attr(
             self.target_polygon) if do_show_tiling_region else ([], [])

        # show cropped region
        super_contour_poly = self.get_super_contour_poly()
        # print('super_contour_poly area:')
        # print(super_contour_poly.area)

        (super_contour_exteriors,
         super_contour_interiors) = BrickLayout.get_polygon_plot_attr(
             super_contour_poly,
             style='lightblue') if do_show_super_contour else ([], [])
        # show selected tiles
        assert self.complete_graph.tiles is not None
        selected_tiles = [
            self.complete_graph.tiles[
                self.inverse_index[i]].get_plot_attribute("yellow")
            for i in range(len(tiles)) if tiles[i] == 1
        ]

        # test = self.get_selected_tiles_union_polygon()
        # print('selected_tiles area:')
        # print(test.area)

        img = plotter.draw_contours(
            tiling_region_exteriors + tiling_region_interiors +
            super_contour_exteriors + super_contour_interiors + selected_tiles,
            file_path=file_name)
        return img

    def show_super_contour(self, plotter: Plotter, file_name):
        super_contour_poly = self.get_super_contour_poly()
        (exteriors_contour_list,
         interiors_list) = BrickLayout.get_polygon_plot_attr(
             super_contour_poly, show_line=True)
        if None not in (plotter, file_name):
            plotter.draw_contours(exteriors_contour_list + interiors_list,
                                  file_name)
        return super_contour_poly

    def show_adjacency_graph(self,
                             save_path,
                             edge_type="all",
                             is_vis_prob=True,
                             node_size=10,
                             edge_width=0.7,
                             xlim=(-1, 1.6),
                             ylim=(-1, 1.6)):
        # create Graph
        G_symmetric = nx.Graph()
        col_edges = [
            tuple(self.collide_edge_index[:, i])
            for i in range(self.collide_edge_index.shape[1])
        ] if self.collide_edge_index.shape[0] > 0 else []
        adj_edges = [
            tuple(self.align_edge_index[:, i])
            for i in range(self.align_edge_index.shape[1])
        ] if self.align_edge_index.shape[0] > 0 else []
        if edge_type == "all":
            edges = col_edges + adj_edges
        elif edge_type == "collision":
            edges = col_edges
        elif edge_type == "adjacent":
            edges = adj_edges
        else:
            print(f"error edge type!!! {edge_type}")

        edge_color = ["gray" for i in range(len(edges))]

        # draw networks
        G_symmetric.add_nodes_from(range(self.node_feature.shape[0]))
        node_color = [
            self.predict_probs[i] if is_vis_prob else "blue"
            for i in range(self.node_feature.shape[0])
        ]
        tile_indices = [
            self.inverse_index[i] for i in range(self.node_feature.shape[0])
        ]
        node_pos_pts = [
            self.complete_graph.tiles[index].tile_poly.centroid
            for index in tile_indices
        ]
        node_pos = list(map(lambda pt: [pt.x, -pt.y], node_pos_pts))
        # print(node_pos)
        # G_symmetric.add_edges_from(edges)

        vmin, vmax = 0.0, 1.0
        cmap = plt.cm.Reds
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        _ = plt.colorbar(sm)
        nx.draw_networkx(G_symmetric,
                         pos=node_pos,
                         node_size=node_size,
                         node_color=node_color,
                         cmap=cmap,
                         width=edge_width,
                         edgelist=edges,
                         edge_color=edge_color,
                         vmin=vmin,
                         vmax=vmax,
                         with_labels=False,
                         style="dashed" if col_edges else "solid")

        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.savefig(save_path, dpi=400)
        print(f'saving file {save_path}...')
        plt.close()

    def show_predict_prob(self, plotter: Plotter, file_name):
        # show prediction probs with color
        predict_probs = self.predict_probs

        min_fill_color, max_fill_color = np.array(
            [255, 255, 255, 50]), np.array([255, 0, 0, 50])
        min_pen_color, max_pen_color = np.array([255, 255, 255, 0]), np.array(
            [127, 127, 127, 255])

        super_contour_poly = self.get_super_contour_poly()
        (exteriors_contour_list,
         interiors_list) = BrickLayout.get_polygon_plot_attr(
             super_contour_poly, show_line=True)

        # sort by prob
        sorted_indices = np.argsort(self.predict_probs)

        plotter.draw_contours(exteriors_contour_list + interiors_list + [
            self.complete_graph.tiles[
                self.inverse_index[i]].get_plot_attribute(
                    (tuple(
                        interp(predict_probs[i],
                               vec1=min_fill_color,
                               vec2=max_fill_color)),
                     tuple(
                         interp(predict_probs[i],
                                vec1=min_pen_color,
                                vec2=max_pen_color)))) for i in sorted_indices
        ],
                              file_path=file_name)

    def get_super_contour_poly(self):
        # return super contour poly if already calculated
        if self.super_contour_poly is None:
            tiles = self.complete_graph.tiles
            selected_tiles = [
                tiles[s].tile_poly.buffer(1e-6) for s in self.re_index.keys()
            ]
            total_polygon = unary_union(selected_tiles).simplify(1e-6)
            self.super_contour_poly = total_polygon
            return total_polygon
        else:
            return self.super_contour_poly

    @staticmethod
    def get_polygon_plot_attr(input_polygon, show_line=False, style=None):
        # return color plot attribute given a shapely polygon
        # return poly attribute with color

        exteriors_contour_list = []
        interiors_list = []

        # set the color for exterior and interiors

        if style is None:
            color = 'light_gray_border' if show_line else 'light_gray'
        else:
            color = (style[0], (127, 127, 127, 0)) if show_line else style

        background_color = 'white_border' if show_line else 'white'

        if isinstance(input_polygon, shapely.geometry.polygon.Polygon):
            exteriors_contour_list = [
                (color, np.array(list(input_polygon.exterior.coords)))
            ]
            interiors_list = [(background_color,
                               np.array(list(interior_poly.coords)))
                              for interior_poly in input_polygon.interiors]

        elif isinstance(input_polygon,
                        shapely.geometry.multipolygon.MultiPolygon):
            exteriors_contour_list = [(color,
                                       np.array(list(polygon.exterior.coords)))
                                      for polygon in input_polygon]
            for each_polygon in input_polygon:
                one_interiors_list = [
                    (background_color, np.array(list(interior_poly.coords)))
                    for interior_poly in each_polygon.interiors
                ]
                interiors_list = interiors_list + one_interiors_list

        return exteriors_contour_list, interiors_list

    def get_selected_tiles(self):
        return [
            self.complete_graph.tiles[self.inverse_index[i]].tile_poly
            for i in range(len(self.predict)) if self.predict[i] == 1
        ]

    def get_selected_tiles_union_polygon(self):
        return unary_union(self.get_selected_tiles())

    def detect_holes(self) -> int:
        # DETECT HOLE
        selected_tiles = [
            self.complete_graph.tiles[self.inverse_index[i]].tile_poly.buffer(
                1e-7) for i in range(len(self.predict)) if self.predict[i] == 1
        ]
        unioned_shape = unary_union(selected_tiles)
        if isinstance(unioned_shape, shapely.geometry.polygon.Polygon):
            # if len(list(unioned_shape.interiors)) > 0:
            #     return True
            return len(list(unioned_shape.interiors))
        elif isinstance(unioned_shape,
                        shapely.geometry.multipolygon.MultiPolygon):
            # if any([
            #         len(list(unioned_shape[i].interiors)) > 0
            #         for i in range(len(unioned_shape))
            # ]):
            #     return True
            return sum([len(list(shape.interiors)) for shape in unioned_shape])
        else:
            raise TypeError("unioned_shape of type {}".format(
                type(unioned_shape)))

    # def _to_torch_tensor(self, device, node_feature, align_edge_index,
    #                      align_edge_features, collide_edge_index,
    #                      collide_edge_features):
    #     x = torch.from_numpy(node_feature).float().to(device)
    #     adj_edge_index = torch.from_numpy(align_edge_index).long().to(device)
    #     adj_edge_features = torch.from_numpy(align_edge_features).float().to(
    #         device)
    #     collide_edge_index = torch.from_numpy(collide_edge_index).long().to(
    #         device)
    #     collide_edge_features = torch.from_numpy(
    #         collide_edge_features).float().to(device)

    #     return (x, adj_edge_index, adj_edge_features, collide_edge_index,
    #             collide_edge_features)

    def get_data_as_torch_tensor(self, device):
        x = self.node_feature.to(
            device
        ) if self.node_feature.device != device else self.node_feature
        adj_edge_index = self.align_edge_index.to(
            device
        ) if self.align_edge_index.device != device else self.align_edge_index
        adj_edge_features = self.align_edge_features
        if self.align_edge_features != device:
            adj_edge_features = self.align_edge_features.to(device).float()
        collide_edge_index = self.collide_edge_index
        if self.collide_edge_index.device != device:
            collide_edge_index = self.collide_edge_index.to(device)
        collide_edge_features = self.collide_edge_features
        if self.collide_edge_features.device != device:
            collide_edge_features = self.collide_edge_features.to(
                device).float()
        return (x, adj_edge_index, adj_edge_features, collide_edge_index,
                collide_edge_features)

    def compute_sub_layout(self, predict):
        assert len(self.node_feature) == len(predict.labelled_nodes) + len(
            predict.unlabelled_nodes)
        sorted_dict = sorted(predict.unlabelled_nodes.items(),
                             key=lambda x: x[0])
        predict.unlabelled_nodes.clear()
        predict.unlabelled_nodes.update(sorted_dict)

        node_mask = torch.zeros(self.node_feature.shape[0], dtype=torch.bool)
        node_mask[torch.tensor(list(predict.unlabelled_nodes.keys()),
                               dtype=torch.long)] = True
        # compute index mapping from original index to current index

        node_re_index = torch.zeros(self.node_feature.shape[0],
                                    dtype=torch.long)
        node_re_index[node_mask] = torch.arange(len(node_re_index[node_mask]))

        # node_re_index = {}
        # for idx, key in enumerate(predict.unlabelled_nodes):
        #     node_re_index[key] = idx

        complete_graph = self.complete_graph
        node_feature = self.node_feature[list(predict.unlabelled_nodes.keys())]

        # index
        ori_coll_edges = self.collide_edge_index
        new_coll_mask = reduce(torch.logical_and, node_mask[ori_coll_edges])
        new_coll_edges = ori_coll_edges[:, new_coll_mask]

        collide_edge_index = node_re_index[new_coll_edges]
        collide_edge_features = self.collide_edge_features[new_coll_mask, :]

        if self.align_edge_index.size(-1):
            ori_align_edges = self.align_edge_index
            new_align_mask = reduce(torch.logical_and,
                                    node_mask[ori_align_edges])
            new_align_edges = ori_align_edges[:, new_align_mask]

            align_edge_index = node_re_index[new_align_edges]
            align_edge_features = self.align_edge_features[new_align_mask, :]
        else:
            align_edge_index = torch.tensor([])
            align_edge_features = torch.tensor([])

        # if collide_edge_index.shape[1] == 0:
        #     collide_edge_index = np.array([])
        #     collide_edge_features = np.array([])
        # if align_edge_index.shape[1] == 0:
        #     align_edge_index = np.array([])
        #     align_edge_features = np.array([])

        # print(node_feature.shape, collide_edge_index.shape,
        #       collide_edge_features.shape, align_edge_index.shape,
        #       align_edge_features.shape)

        # compute index mapping from new index to original index
        node_inverse_index = {}
        for idx, key in enumerate(predict.unlabelled_nodes):
            node_inverse_index[idx] = key

        fixed_re_index = {}
        for i in range(node_feature.shape[0]):
            try:
                fixed_re_index[self.inverse_index[node_inverse_index[i]]] = i
            except KeyError as e:
                print("key error", e)
                print("node id {}, node inverse index {}".format(
                    i, node_inverse_index[i]))
                raise e

        return BrickLayout(
            complete_graph,
            node_feature,
            collide_edge_index,
            collide_edge_features,
            align_edge_index,
            align_edge_features,
            fixed_re_index,
            target_polygon=self.target_polygon), node_inverse_index

    def update_tiles(self):
        tiles = self.complete_graph.tiles
        selected_indices = [k for k in self.re_index.keys()]
        selected_tiles = [tiles[s] for s in selected_indices]
        self.tiles = selected_tiles

    @staticmethod
    def assert_equal_layout(brick_layout_1, brick_layout_2):
        assert np.array_equal(brick_layout_1.node_feature,
                              brick_layout_2.node_feature)
        assert np.array_equal(brick_layout_1.collide_edge_index,
                              brick_layout_2.collide_edge_index)
        assert np.array_equal(brick_layout_1.collide_edge_features,
                              brick_layout_2.collide_edge_features)
        assert np.array_equal(brick_layout_1.align_edge_index,
                              brick_layout_2.align_edge_index)
        assert np.array_equal(brick_layout_1.align_edge_features,
                              brick_layout_2.align_edge_features)

        # mapping from index of complete graph to index of super graph
        for key in brick_layout_1.re_index.keys():
            assert brick_layout_1.re_index[key] == brick_layout_2.re_index[key]
        for key in brick_layout_2.re_index.keys():
            assert brick_layout_2.re_index[key] == brick_layout_1.re_index[key]

        # assert others
        assert np.array_equal(brick_layout_1.predict, brick_layout_2.predict)
        assert np.array_equal(brick_layout_1.predict_probs,
                              brick_layout_2.predict_probs)
        assert brick_layout_1.predict_order == brick_layout_2.predict_order
        assert brick_layout_1.target_polygon == brick_layout_2.target_polygon
