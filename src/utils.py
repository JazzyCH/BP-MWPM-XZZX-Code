import contextlib
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import math
import numpy as np
import networkx as nx
import src.globals as g


def nx_2D_lattice_color(G, edge_label, flipped):
    # plt.figure(figsize=(300, 9))
    # plt.figure(figsize=(6, 6))
    pos = {}
    for node in G.nodes:
        pos[node] = (node[2], -node[1])

    colors = nx.get_edge_attributes(G,'color').values()

    nx.draw(G, pos,
            edge_color=colors,
            # width=list(weights),
            with_labels=False)
    labels = nx.get_edge_attributes(G, edge_label)
    labels = {x: round(y, 3) for (x, y) in labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    paulis = [None, "X", "Y", "Z"]
    for error in [g.ERROR_X, g.ERROR_Y, g.ERROR_Z]:
       for node in flipped[error]:
            r = -node[0]
            c = node[1]
            # plt.scatter(c, r, color="g", s=100)
            plt.text(c, r, paulis[error], fontsize="xx-large", fontweight="bold")

    plt.show()



def nx_2D_lattice(G, edge_label):
    plt.figure(figsize=(6, 6))
    pos = {}
    for node in G.nodes:
        pos[node] = (node[2], -node[1])
    nx.draw_networkx(G, pos)
    labels = nx.get_edge_attributes(G, edge_label)
    labels = {x: round(y, 3) for (x, y) in labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


def base_2D_lattice(syn_nodes, virtual):
    # plt.figure(figsize=(300,9))
    colors = [
        ["b", "cornflowerblue"],
        ["r", "lightcoral"]
    ]

    for subgraph in [g.GROUP_A, g.GROUP_B]:
        for node in syn_nodes[subgraph]:
            r = -node[1]
            c = node[2]
            plt.scatter(c, r, color=colors[subgraph][0])
            plt.text(c, r, str(node))

        for node in virtual[subgraph]:
            r = -node[1]
            c = node[2]
            plt.scatter(c, r, color=colors[subgraph][1])
            plt.text(c, r, str(node))
    plt.show()


def err_syndrome_graph(syn_nodes, virtual, err_syndrome, flipped):
    colors = [
        ["b", "cornflowerblue"],
        ["r", "lightcoral"]
    ]

    for subgraph in [g.GROUP_A, g.GROUP_B]:
        for node in syn_nodes[subgraph]:
            r = -node[1]
            c = node[2]
            plt.scatter(c, r, color=colors[subgraph][0])
            plt.text(c, r, str(node))

        for node in virtual[subgraph]:
            r = -node[1]
            c = node[2]
            plt.scatter(c, r, color=colors[subgraph][1])
            plt.text(c, r, str(node))

    paulis = [None, "X", "Y", "Z"]
    for error in [g.ERROR_X, g.ERROR_Y, g.ERROR_Z]:
       for node in flipped[error]:
            r = -node[0]
            c = node[1]
            plt.scatter(c, r, color="g", s=100)
            plt.text(c, r, paulis[error], fontsize="xx-large", fontweight="bold")

    for subgraph in err_syndrome:
        for node in subgraph:
            r = -node[1]
            c = node[2]
            plt.scatter(c, r, color="orange", s=100)

    plt.show()



# https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
