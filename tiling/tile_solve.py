from typing import List, Tuple, Union
import pickle
import logging
import os
# from shapely.geometry import Polygon
import numpy as np

from utils.shape_processor import load_polygons
from tiling.tile_factory import crop_multiple_layouts_from_contour
from tiling.brick_layout import BrickLayout
from tiling.tile_graph import TileGraph

from interfaces.qt_plot import Plotter
from solver.MIS.base_solver import BaseSolver
# from utils.data_util import load_bricklayout, write_bricklayout

EPS = 1e-5


def tile_silhouette_list(solver: BaseSolver, complete_graph: TileGraph,
                         silhouette_list: Union[List[str], str], root_path):
    plotter = Plotter()

    if type(silhouette_list) == str:
        silhouette_list = list(np.genfromtxt(silhouette_list, dtype=str))

    for silhouette_path in silhouette_list:
        logging.info(silhouette_path)
        tiling_a_region(plotter, complete_graph, solver, silhouette_path,
                        root_path)
        # name = os.path.split(os.path.splitext(silhouette_path)[0])[-1]
        # get_cropped_layouts(complete_graph, silhouette_path)
        # logging.info(name)


def get_cropped_layouts(complete_graph, silhouette_path):
    pre_generated_path = os.path.splitext(silhouette_path)[0] + '.pkl'
    cropped_brick_layouts: List[Tuple[BrickLayout, float]]
    try:
        cropped_brick_layouts = pickle.load(open(pre_generated_path, 'rb'))
    except FileNotFoundError:
        exterior_contour, interior_contours = load_polygons(silhouette_path)
        # plot the select tiling region
        # base_polygon = Polygon(exterior_contour, holes=interior_contours)
        # exteriors_contour_list, \
        #     interiors_list = BrickLayout.get_polygon_plot_attr(
        #             base_polygon, show_line=True)
        # plotter.draw_contours(
        #     debugger.file_path(f'tiling_region_{silhouette_file_name[:-4]}.png'),
        #     exteriors_contour_list + interiors_list)

        # get candidate tile placements inside the tiling region by cropping
        cropped_brick_layouts = crop_multiple_layouts_from_contour(
            exterior_contour,
            interior_contours,
            complete_graph=complete_graph,
            start_angle=0,
            end_angle=30,
            num_of_angle=1,
            movement_delta_ratio=[0, 0.5],
            margin_padding_ratios=[0.5])

        pickle.dump(cropped_brick_layouts, open(pre_generated_path, 'wb'))
    return cropped_brick_layouts


def tiling_a_region(plotter: Plotter,
                    complete_graph: TileGraph,
                    solver: BaseSolver,
                    silhouette_path: str,
                    root_path: str,
                    logger_name: str = "TILING"):
    # environment.complete_graph.show_complete_super_graph(
    #     plotter, "complete_graph.png")
    cropped_brick_layouts = get_cropped_layouts(
        complete_graph=complete_graph, silhouette_path=silhouette_path)
    # show the cropped tile placements
    # for idx, (brick_layout, coverage) in enumerate(cropped_brick_layouts):
    #     brick_layout.show_candidate_tiles(
    #         plotter,
    #         os.path.join(root_path, f"candi_tiles_{idx}_{coverage}.png"))

    # tiling solving
    solutions: List[Tuple[BrickLayout, float]] = []
    for idx, (queried_brick_layout, _) in enumerate(cropped_brick_layouts):
        print(f"solving cropped brick layout {idx} :")

        # direct solve (origin)
        # result_brick_layout, score = solver.solve(result_brick_layout)

        # find best solution in 20 trials
        result_brick_layout, score = solver.solve_with_trials(
            queried_brick_layout, 3)
        solutions.append((result_brick_layout, score))

    # show solved layout
    for idx, solved_layout in enumerate(solutions):
        result_brick_layout, score = solved_layout

        # hacking for probs
        result_brick_layout.predict_probs = result_brick_layout.predict

        # write_bricklayout(folder_path=os.path.join(root_path, "./"),
        #                   file_name=f'{score}_{idx}_data.pkl',
        #                   brick_layout=result_brick_layout,
        #                   with_features=False)

        # reloaded_layout = load_bricklayout(file_path=os.path.join(
        #     root_path, f'{score}_{idx}_data.pkl'),
        #                                    complete_graph=complete_graph)

        # # asserting correctness
        # BrickLayout.assert_equal_layout(result_brick_layout, reloaded_layout)

        # calculate metric (testing, just ignore)
        # metric = result_brick_layout.get_selected_tiles_union_polygon(
        # ).area / result_brick_layout.get_super_contour_poly().area
        # print('----------')
        # print('metrc is:')
        # print(metric)

        name = os.path.split(os.path.splitext(silhouette_path)[0])[-1]
        result_brick_layout.show_predict(
            plotter,
            os.path.join(root_path, f'{name}_{score}_{idx}_predict.png'),
            do_show_super_contour=True,
            do_show_tiling_region=True)
        # result_brick_layout.show_super_contour(
        #     plotter, os.path.join(root_path,
        #                           f'{score}_{idx}_super_contour.png'))
        # result_brick_layout.show_adjacency_graph(
        #     os.path.join(root_path, f'{score}_{idx}_vis_graph.png'))
        # result_brick_layout.show_predict_prob(
        #     plotter,
        #     os.path.join(root_path, f'{name}_{score}_{idx}_prob.png'))
