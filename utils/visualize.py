import networkx as nx
import numpy as np


def draw_full_graph(g, probs=None, ax=None):
    """
    g:networkx graph
    probs:length N numpy array
    x: matplotlib axes Subplot object
    """

    green = np.array([0.0,0.1,0.0])
    red = np.array([1.0,0.0,0.0])
    blue = np.array([0.0,0.0,1.0])

    node_colors = None
    edge_colors = None

    if probs:
        assert len(probs.shape) == 1
        assert np.all(0.0<=probs<=1.0)

        node_colors = [green * prob for prob in probs]

        edge_probs = [probs[i]*probs[j] for i,j in g.edges]
        edge_colors = [blue + (red - blue)*prob for prob in edge_probs]

    pos = None

    nx.draw_networkx(
        g,
        with_labels=False,
        arrows=False,
        arrowsize=3,
        node_size=30,
        pos=pos,
        edge_color=edge_colors,
        ax=ax,
        node_color=node_colors
    )