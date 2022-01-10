from typing import Tuple, List
import torch
import numpy as np
# import os
from torch_geometric.utils import subgraph

from sampler.base_sampler import Sampler
from solver.MIS.base_solver import BaseSolver
from tiling.brick_layout import BrickLayout
from interfaces.qt_plot import Plotter
from utils.crop import crop_2d_circle

plotter = Plotter()


def solve_by_crop(
    full_graph: BrickLayout,
    model: torch.nn.Module,
    solver: BaseSolver,
    sampler: Sampler,
    show_intermediate: bool = False,
    log_dir: str = None,
) -> Tuple[BrickLayout, torch.Tensor, List]:
    r"""
    full_graph: original full graph
    show_intermediate: if true, intermediate_results are returned,
        else an empty list is returned

    return:
        final result layout: same as :obj:`full_graph`, but
            with :obj:`predict` set as solution mask
        prob_records: 1d tensor of prob of nodes when they
            are selected in the sampled graph
        intermediate_results: list of np arrays, corresponding
            to :obj:`predict` in final result layout
    """
    '''
    full_graph:
        original full graph
    sampled_edges:
        2 * |E| of sampled graph
    sampled_node_ids:
        length = |V| of sampled graph
        mapping to node ids in full graph
    current_full_graph:
        current version of full graph
    current_node_ids:
        length = current |V|
        current version of sampled graph
        mapping to node ids in current full graph
    current_edges:
        2 * current |E|
        current version of sampled graph
        node id relabelled from 0 to current |V|
    '''
    # make dir
    # tile_path = os.path.join(log_dir, 'candidate_tiles')
    # if not os.path.exists(tile_path):
    #     os.mkdir(tile_path)
    # step_path = os.path.join(log_dir, 'steps')
    # if not os.path.exists(step_path):
    #     os.mkdir(step_path)

    # full_tiles = full_graph.show_candidate_tiles(
    #     plotter, os.path.join(log_dir, 'candidate_tiles.png'))
    # full_graph.tiles = full_tiles
    full_graph.update_tiles()

    try:
        sampled_edges, sampled_node_mask = sampler(
            full_graph.collide_edge_index)
    except RuntimeError as e:
        print("data: ", getattr(full_graph, "idx", "unknown"))
        raise e
    sampled_node_ids = torch.arange(
        full_graph.node_feature.size(0))[sampled_node_mask]
    current_full_graph = full_graph
    current_node_ids = sampled_node_ids
    current_edges = sampled_edges

    step = 0
    prob_records = []
    final_complete_solution = []
    intermediate_results = []

    # iteratively process
    while len(current_node_ids) > 0:
        '''
        special case as batch norm in network requires more
        than 1 channels when training
        '''
        if current_node_ids.size(0) == 1:
            selected_node = torch.tensor(0)
        else:
            probs = model(x=current_full_graph.node_feature[current_node_ids,
                                                            -1:],
                          col_e_idx=current_edges)
            prob_records.append(torch.max(probs).unsqueeze(0))
            selected_node = torch.argmax(probs)
        original_node_id = current_node_ids[selected_node]

        current_full_edges = current_full_graph.collide_edge_index
        # print(len(current_full_graph.tiles))
        (_, collide_edge_index, collide_edge_features, _, _, _,
         nodes) = crop_2d_circle(original_node_id.item(),
                                 current_full_graph,
                                 20,
                                 low=0.5,
                                 high=0.7)
        if nodes is None:
            break

        # reindex
        col, row = current_full_graph.collide_edge_index
        num_nodes = current_full_graph.node_feature.size(0)
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[nodes] = torch.arange(nodes.size(0), device=row.device)
        collide_edge_index = node_idx[collide_edge_index]

        queried_subgraph = BrickLayout(
            complete_graph=full_graph.complete_graph,
            node_feature=current_full_graph.node_feature[nodes],
            collide_edge_index=collide_edge_index,
            collide_edge_features=collide_edge_features,
            align_edge_index=torch.tensor([[], []]),
            align_edge_features=torch.tensor([[]]),
            re_index={
                current_full_graph.inverse_index[k.item()]: i
                for i, k in enumerate(nodes)
            })

        # solve cropped subgraph
        result, score = solver.solve(queried_subgraph)

        # result.predict_probs = result.predict
        # result.show_predict(plotter,
        #                     os.path.join(step_path,
        #                                  f'step_{step}_{score}_predict.png'),
        #                     do_show_super_contour=True,
        #                     do_show_tiling_region=True)

        # selected node ids in complete graph
        # keep this for final reward calculation
        solution_complete_ids = [
            result.inverse_index[n.item()]
            for n in torch.arange(queried_subgraph.node_feature.size(0))[
                torch.tensor(result.predict, dtype=torch.bool)]
        ]

        final_complete_solution += solution_complete_ids
        if show_intermediate:
            temp_solution = np.zeros(full_graph.node_feature.size(0),
                                     dtype=np.int32)
            for i in final_complete_solution:
                temp_solution[full_graph.re_index[i]] = 1
            intermediate_results.append(temp_solution)

        # solution: selected node ids in current full graph
        solution = torch.tensor([
            current_full_graph.re_index[complete_graph_node]
            for complete_graph_node in solution_complete_ids
        ])

        # marked_nodes: marked out nodes in current full graph
        # marked_edges: marked out edges in current full graph
        marked_nodes = torch.zeros(current_full_graph.node_feature.size(0),
                                   dtype=torch.bool)
        marked_nodes[solution] = True

        marked_edges = torch.logical_or(marked_nodes[current_full_edges[0]],
                                        marked_nodes[current_full_edges[1]])

        marked_nodes[current_full_edges[0][marked_edges]] = True
        marked_nodes[current_full_edges[1][marked_edges]] = True

        # update current_full_graph based on marked nodes and edges
        remaining_nodes = torch.logical_not(marked_nodes)
        new_full_nodes = current_full_graph.node_feature[remaining_nodes]
        new_full_edges, new_full_edge_features = subgraph(
            remaining_nodes,
            current_full_edges,
            current_full_graph.collide_edge_features,
            relabel_nodes=True,
            num_nodes=current_full_graph.node_feature.size(0))

        full_node_prev_to_curr = torch.empty_like(remaining_nodes,
                                                  dtype=torch.long)
        full_node_prev_to_curr[remaining_nodes] = torch.arange(
            remaining_nodes.sum())

        # print(full_node_prev_to_curr[remaining_nodes])
        current_full_graph = BrickLayout(
            complete_graph=current_full_graph.complete_graph,
            node_feature=new_full_nodes,
            collide_edge_index=new_full_edges,
            collide_edge_features=new_full_edge_features,
            align_edge_index=torch.tensor([[], []]),
            align_edge_features=torch.tensor([[]]),
            re_index={
                k: full_node_prev_to_curr[v].item()
                for k, v in current_full_graph.re_index.items()
                if remaining_nodes[v]
            })
        # full_tiles = current_full_graph.show_candidate_tiles(
        #     plotter, os.path.join(tile_path,
        #                           f'step_{step}_candidate_tiles.png'))
        # current_full_graph.tiles = full_tiles
        current_full_graph.update_tiles()

        # update current_node_ids and current_edges
        remaining_sampled_node_mask = remaining_nodes[current_node_ids]
        current_node_ids = full_node_prev_to_curr[
            current_node_ids[remaining_sampled_node_mask]]
        current_edges, _ = subgraph(
            remaining_sampled_node_mask,
            current_edges,
            relabel_nodes=True,
            num_nodes=remaining_sampled_node_mask.size(0))
        step += 1

    # final solve
    # result, score = solver.solve(current_full_graph)
    # result.predict_probs = result.predict
    # result.show_predict(plotter,
    #                     os.path.join(step_path,
    #                                  f'step_{step}_{score}_predict.png'),
    #                     do_show_super_contour=True,
    #                     do_show_tiling_region=True)
    # solution_complete_ids = [
    #     result.inverse_index[n.item()]
    #     for n in torch.arange(current_full_graph.node_feature.size(0))[
    #         torch.tensor(result.predict, dtype=torch.bool)]
    # ]

    # final_complete_solution += solution_complete_ids
    # if show_intermediate:
    #     temp_solution = np.zeros(full_graph.node_feature.size(0),
    #                              dtype=np.int32)
    #     for i in final_complete_solution:
    #         temp_solution[full_graph.re_index[i]] = 1
    #     intermediate_results.append(temp_solution)

    # final output
    final_result_layout = full_graph
    final_solution = np.zeros(final_result_layout.node_feature.size(0),
                              dtype=np.int32)
    for i in final_complete_solution:
        final_solution[final_result_layout.re_index[i]] = 1
    final_result_layout.predict = final_solution

    return final_result_layout, torch.cat(prob_records,
                                          dim=0), intermediate_results
