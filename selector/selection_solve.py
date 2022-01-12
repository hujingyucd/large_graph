from typing import Tuple, List
import torch
import numpy as np
from torch_geometric.utils import subgraph

from sampler.base_sampler import Sampler
from solver.MIS.base_solver import BaseSolver
from tiling.brick_layout import BrickLayout
from utils.crop import crop_2d_circle


def solve_by_sample_selection(
        full_graph: BrickLayout,
        model: torch.nn.Module,
        solver: BaseSolver,
        sampler: Sampler,
        show_intermediate: bool = False
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

    prob_records = []
    final_complete_solution = []
    intermediate_results = []
    # iteratively process
    while len(current_node_ids) > 0:
        # print(current_edges.size())
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
            selected_node = torch.multinomial(probs.squeeze(), 1)
            prob_records.append(probs[selected_node])

        original_node_id = current_node_ids[selected_node]

        # get BFS subgraph from current full graph
        current_full_edges = current_full_graph.collide_edge_index
        # print("current full edges", current_full_edges.size())
        # sub_nodes, sub_edges, _, edge_masks = k_hop_subgraph(
        #     original_node_id.item(),
        #     3,
        #     current_full_edges,
        #     relabel_nodes=True,
        #     num_nodes=current_full_graph.node_feature.size(0))
        sub_nodes = crop_2d_circle(original_node_id.item(),
                                   current_full_graph,
                                   20,
                                   low=0.5,
                                   high=0.7)
        assert sub_nodes is not None
        print(len(sub_nodes))
        sub_edges, sub_edge_features = subgraph(
            sub_nodes,
            current_full_edges,
            current_full_graph.collide_edge_features,
            relabel_nodes=True,
            num_nodes=current_full_graph.node_feature.size(0))

        queried_subgraph = BrickLayout(
            complete_graph=full_graph.complete_graph,
            node_feature=current_full_graph.node_feature[sub_nodes],
            collide_edge_index=sub_edges,
            collide_edge_features=sub_edge_features,
            align_edge_index=torch.tensor([[], []]),
            align_edge_features=torch.tensor([[]]),
            re_index={
                current_full_graph.inverse_index[k.item()]: i
                for i, k in enumerate(sub_nodes)
            })

        # solve BFS subgraph
        result, _ = solver.solve(queried_subgraph)

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
        # marked edges: marked out edges in current full graph
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

    final_result_layout = full_graph
    final_solution = np.zeros(final_result_layout.node_feature.size(0),
                              dtype=np.int32)
    for i in final_complete_solution:
        final_solution[final_result_layout.re_index[i]] = 1
    final_result_layout.predict = final_solution

    return final_result_layout, torch.cat(prob_records,
                                          dim=0), intermediate_results
