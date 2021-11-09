from typing import List, Tuple, Union
import logging
import os
# from shapely.geometry import Polygon
import numpy as np

from utils.shape_processor import load_polygons
from utils.data_util import load_bricklayout
from tiling.tile_factory import crop_multiple_layouts_from_contour
from tiling.tile_factory import save_all_layout_info
from tiling.brick_layout import BrickLayout
from tiling.tile_graph import TileGraph

from interfaces.qt_plot import Plotter
from solver.MIS.base_solver import BaseSolver
# from utils.data_util import load_bricklayout, write_bricklayout

EPS = 1e-5


def tile_silhouette_list(solver: BaseSolver,
                         complete_graph: TileGraph,
                         silhouette_list: Union[List[str], str],
                         save_path,
                         cropped_layouts_dir: str = ""):
    plotter = Plotter()

    if type(silhouette_list) == str:
        silhouette_list = list(np.genfromtxt(silhouette_list, dtype=str))

    for silhouette_path in silhouette_list:
        logging.info(silhouette_path)
        tiling_a_region(plotter, complete_graph, solver, silhouette_path,
                        save_path, cropped_layouts_dir)


def crop_layouts(complete_graph: TileGraph, silhouette_path):
    exterior_contour, interior_contours = load_polygons(silhouette_path)
    cropped_brick_layouts = crop_multiple_layouts_from_contour(
        exterior_contour,
        interior_contours,
        complete_graph=complete_graph,
        start_angle=0,
        end_angle=30,
        num_of_angle=1,
        movement_delta_ratio=[0, 0.5],
        margin_padding_ratios=[0.8])
    return [tup[0] for tup in cropped_brick_layouts]


def get_cropped_layouts(complete_graph: TileGraph,
                        silhouette_path,
                        cropped_layouts_dir=""):
    if cropped_layouts_dir:
        silhouette_name = os.path.splitext(
            os.path.split(silhouette_path)[-1])[0]
        info_path = os.path.join(cropped_layouts_dir, silhouette_name + ".txt")
        try:
            with open(info_path, 'r') as f:
                num_layout = int(f.readline().split()[-1])
            cropped_brick_layouts = [
                load_bricklayout(os.path.join(
                    cropped_layouts_dir,
                    silhouette_name + "_{}_data.pkl".format(i)),
                                 complete_graph=complete_graph)
                for i in range(num_layout)
            ]
        except FileNotFoundError:
            try:
                os.makedirs(cropped_layouts_dir)
            except FileExistsError:
                pass
            cropped_brick_layouts = crop_layouts(complete_graph,
                                                 silhouette_path)
            for i, layout in enumerate(cropped_brick_layouts):
                save_all_layout_info("{}_{}".format(silhouette_name, i),
                                     layout, cropped_layouts_dir, True)
            with open(info_path, 'w') as f:
                f.write("layout_number: {}".format(len(cropped_brick_layouts)))
    else:
        cropped_brick_layouts = crop_layouts(complete_graph, silhouette_path)
    return cropped_brick_layouts


def tiling_a_region(plotter: Plotter,
                    complete_graph: TileGraph,
                    solver: BaseSolver,
                    silhouette_path: str,
                    save_path: str,
                    cropped_layouts_dir: str = "",
                    logger_name: str = "TILING"):
    # environment.complete_graph.show_complete_super_graph(
    #     plotter, "complete_graph.png")
    cropped_brick_layouts = get_cropped_layouts(
        complete_graph=complete_graph,
        silhouette_path=silhouette_path,
        cropped_layouts_dir=cropped_layouts_dir)
    print(len(cropped_brick_layouts))
    # show the cropped tile placements
    # for idx, (brick_layout, coverage) in enumerate(cropped_brick_layouts):
    #     brick_layout.show_candidate_tiles(
    #         plotter,
    #         os.path.join(save_path, f"candi_tiles_{idx}_{coverage}.png"))

    # tiling solving
    solutions: List[Tuple[BrickLayout, float]] = []
    for idx, queried_brick_layout in enumerate(cropped_brick_layouts):
        print(f"solving cropped brick layout {idx} :")

        # direct solve (origin)
        # result_brick_layout, score = solver.solve(result_brick_layout)

        # find best solution in multiple trials
        result_brick_layout, score = solver.solve_with_trials(
            queried_brick_layout, 3)
        solutions.append((result_brick_layout, score))

    # show solved layout
    for idx, solved_layout in enumerate(solutions):
        result_brick_layout, score = solved_layout

        # hacking for probs
        result_brick_layout.predict_probs = result_brick_layout.predict
        print("holes: ", result_brick_layout.detect_holes())

        # write_bricklayout(folder_path=os.path.join(save_path, "./"),
        #                   file_name=f'{score}_{idx}_data.pkl',
        #                   brick_layout=result_brick_layout,
        #                   with_features=False)

        # reloaded_layout = load_bricklayout(file_path=os.path.join(
        #     save_path, f'{score}_{idx}_data.pkl'),
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
            os.path.join(save_path, f'{name}_{score}_{idx}_predict.png'),
            do_show_super_contour=True,
            do_show_tiling_region=True)
        # result_brick_layout.show_super_contour(
        #     plotter, os.path.join(save_path,
        #                           f'{name}_{score}_{idx}_super_contour.png'))
        # result_brick_layout.show_adjacency_graph(
        #     os.path.join(save_path, f'{name}_{score}_{idx}_vis_graph.png'))
        # result_brick_layout.show_predict_prob(
        #     plotter,
        #     os.path.join(save_path, f'{name}_{score}_{idx}_prob.png'))
