import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch


def draw_full_graph(g, probs=None, ax=None):
    """
    g:networkx graph
    probs:length N numpy array
    x: matplotlib axes Subplot object
    """

    green = np.array([0.0, 0.1, 0.0])
    red = np.array([1.0, 0.0, 0.0])
    blue = np.array([0.0, 0.0, 1.0])

    node_colors = None
    edge_colors = None

    if probs:
        assert len(probs.shape) == 1
        assert np.all(0.0 <= probs <= 1.0)

        node_colors = [green * prob for prob in probs]

        edge_probs = [probs[i] * probs[j] for i, j in g.edges]
        edge_colors = [blue + (red - blue) * prob for prob in edge_probs]

    pos = None

    nx.draw_networkx(g,
                     with_labels=False,
                     arrows=False,
                     arrowsize=3,
                     node_size=30,
                     pos=pos,
                     edge_color=edge_colors,
                     ax=ax,
                     node_color=node_colors)


def draw_sample_graph(data, probs=None, root=0, depth=1, ax=None):
    """
    data: raw graph data
    probs: cuda tensor probs
    root: root node for BFS
    depth: depth limit for BFS
    ax: matplotlib AxesSubplot object
    """
    # convert probs to length N numpy array
    probs = probs.cpu().detach().numpy()
    probs = np.transpose(probs)
    probs = probs[0]

    # convert data to edges
    src = data.edge_index[0].cpu().numpy()
    dst = data.edge_index[1].cpu().numpy()
    edgelist = zip(src, dst)

    # construct full graph from edges
    g = nx.OrderedGraph()
    for i, j in edgelist:
        g.add_edge(i, j)
    '''plt.figure(figsize=(30, 14))
    nx.draw_networkx(g)
    plt.savefig('{}.png'.format('FullGraph'))'''

    # BFS from selected root node (DEFAULT at node 0, depth 1)
    edges_traverse = list(nx.traversal.bfs_edges(g, root, depth_limit=depth))
    nodes = [root] + [v for u, v in edges_traverse]
    nodes.sort()

    # construct sample graph from traversed edges
    G = nx.OrderedGraph()
    for i, j in list(edges_traverse):
        G.add_edge(i, j)

    # collect remaining edges from full graph
    for i, j in g.edges:
        if (i in nodes) and (j in nodes):
            G.add_edge(i, j)

    # selection probability of selected nodes
    probs_selected = [probs[index] for index in nodes]

    # sort nodes by probs
    mapping = zip(probs_selected, nodes)
    nodes_sorted_by_probs = [x for _, x in sorted(mapping)]
    probs_sorted = [probs[index] for index in nodes_sorted_by_probs]

    # relabel the nodes in G with new node order (For gradient color in circular graph)
    nodes_mapping = dict(zip(nodes, nodes_sorted_by_probs))
    G = nx.relabel_nodes(G, nodes_mapping)

    # plot and save graph
    plt.figure(figsize=(30, 14))
    pos = None
    # circular layout
    '''pos = nx.circular_layout(G)
    pos[0] = np.array([0, 0])'''

    nx.draw_networkx(G,
                     pos=pos,
                     with_labels=False,
                     edge_color='#d5d6de',
                     node_color=probs_sorted,
                     cmap=plt.cm.Blues,
                     node_size=30)
    plt.savefig('{}.png'.format('graph'))
