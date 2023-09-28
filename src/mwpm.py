import networkx as nx
import numpy as np
import math
import os
import ctypes
from pathlib import Path
from scipy.special import binom
from itertools import combinations
from numba import njit

import src.globals as g


SCRIPT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
_libpypm = ctypes.cdll.LoadLibrary(Path(SCRIPT_DIR.parents[0], "bin", "libpypm.so"))
_libpypm.infty.argtypes = None
_libpypm.infty.restype = ctypes.c_int
_libpypm.mwpm.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                          ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
_libpypm.mwpm.restype = None
# Integer that represents infinity (Blossom V algorithm). (Can be useful when converting float to int weights for MWPM)
INFTY = _libpypm.infty()


# Scale the distance for MWPM function requires int(distance)
def _scale_edge_weights(err_sub):
    max_abs_wt = max(abs(x[2]) for x in err_sub)
    if max_abs_wt != 0:
        scaling = INFTY / 10 / max_abs_wt
    else:
        scaling = INFTY
    for i in range(len(err_sub)):
        err_sub[i][2] = int(err_sub[i][2] * scaling)


# Append virtual node to make the number of nodes even
def _append_mwpm_virtual_node(err_sub, virt_sub, num_node):
    if num_node % 2:
        for source in virt_sub:
            err_sub.append([source, (0, -1, -1), 0])


def blossom_path_summation(virtual_sub, source, target, subgraph, is_depolarized):
    """
    Calculates the degeneracy between nodes.
    Args:
        source (tuple): The starting syndrome/virtual node
        target (tuple): The ending syndrome/virtual node
        subgraph (g.GROUP_A or g.GROUP_B): Specifying which subgraph the nodes are supposed to be on
    Returns:
        degeneracy (int): degeneracy between nodes.
    """
    # TODO OPTIMIZE: CAN AT LEAST CUT ITERATION BY PASSING AN EXTRA ARGUMENT OF MATCHED VIRTUAL AND SYNDROME PAIR
    # If one of the nodes is virtual, check degeneracy of the other with all virtual nodes
    match = None
    if source[0] == 0:
        match = target
        matched = source
    elif target[0] == 0:
        match = source
        matched = target

    z_edge = abs(int(target[2] - source[2]))
    x_edge = abs(int(target[1] - source[1]))

    if match:
        if x_edge == 0 or z_edge == 0:
            degeneracy = 1
        elif subgraph == g.GROUP_A:
            degeneracy = int(binom((z_edge + x_edge - 1), z_edge))
        else:
            degeneracy = int(binom((z_edge + x_edge - 1), x_edge))

        for virtual_node in virtual_sub:
            if virtual_node != matched:
                z_edge_n = abs(int(virtual_node[2] - match[2]))
                x_edge_n = abs(int(virtual_node[1] - match[1]))

                if (
                    (is_depolarized and (z_edge_n + x_edge_n) == (z_edge + x_edge))
                    or (not is_depolarized and z_edge == z_edge_n and x_edge == x_edge_n)
                ):
                    if x_edge_n == 0 or z_edge_n == 0:
                        degeneracy += 1
                    elif subgraph == g.GROUP_A:
                        degeneracy += int(binom((z_edge_n + x_edge_n - 1), z_edge_n))
                    else:
                        degeneracy += int(binom((z_edge_n + x_edge_n - 1), x_edge_n))
    else:
        degeneracy = int(binom((z_edge + x_edge), z_edge))

    return degeneracy


def blossom_error_graph(
        virtual, err_syndrome, err_probs,
        z_edge_weight, x_edge_weight,
        is_depolarized, multi=False
    ):
    """
    Construct error graph of subgraph A and B consisting of (node_a, node_n, distance)
    for MWPM. Multi-path summation optional. Distance calculated in self._path_summation(...),
    degeneracy calculated in self._degeneracy_cal.
    Args:
        err_syndrome: Dictionary containing the nodes for subgraph A and B separately
        multi: True for multi-path summation, False ignores such degeneracy
    Returns:
        error_graph: Dictionary containing the error graph ready for MWPM for subgraph A and B separately
    """
    px = err_probs[1]
    py = err_probs[2]
    pz = err_probs[3]

    error_graph = [[], []]
    for subgraph in [g.GROUP_A, g.GROUP_B]:
        nodes = err_syndrome[subgraph] + virtual[subgraph] # All the nodes that can be matched
        lo = len(nodes)

        for source, target in combinations(nodes, 2):
            # Distance between virtual nodes is 0
            if source[0] == 0 and target[0] == 0:
                distance = 0
            else:
                z_edge = abs(int(target[2] - source[2]))
                x_edge = abs(int(target[1] - source[1]))

                if px == 0 and py == 0 and x_edge != 0:
                    continue
                elif pz == 0 and py == 0 and z_edge != 0:
                    continue
                else:
                    distance = z_edge * z_edge_weight + x_edge * x_edge_weight

                if multi:
                    degeneracy = blossom_path_summation(virtual[subgraph], source, target, subgraph, is_depolarized)
                    distance -= math.log(degeneracy)

            error_graph[subgraph].append([source, target, distance])

        _append_mwpm_virtual_node(error_graph[subgraph], virtual[subgraph], lo)
        _scale_edge_weights(error_graph[subgraph])

    return error_graph


@njit(cache=True)
def nx_path_summation(
        num_node, num_err, s_ind, t_ind, s_is_virtual, t_is_virtual,
        distance_arr, degeneracy_arr,
        atol=1e-5
    ):
    """
    Calculates the degeneracy between nodes.
    Args:
        source (tuple): The starting syndrome/virtual node
        target (tuple): The ending syndrome/virtual node
        subgraph (g.GROUP_A or g.GROUP_B): Specifying which subgraph the nodes are supposed to be on
    Returns:
        degeneracy (int): degeneracy between nodes.
    """
    degeneracy = degeneracy_arr[s_ind, t_ind]

    is_match = False
    if s_is_virtual:
        match_ind = t_ind
        matched_ind = s_ind
        is_match = True
    elif t_is_virtual:
        match_ind = s_ind
        matched_ind = t_ind
        is_match = True

    if is_match:
        shortest_distance = distance_arr[s_ind, t_ind]

        for v_ind in range(num_err, num_node):
            if v_ind != matched_ind:
                distance = distance_arr[match_ind, v_ind]
                if abs(distance - shortest_distance) < atol:
                    degeneracy += degeneracy_arr[match_ind, v_ind]

    return degeneracy


def nx_error_graph(S, virtual, err_syndrome, multi=False):
    """
    Construct error graph of subgraph A and B consisting of (node_a, node_n, distance)
    for MWPM. Multi-path summation optional. Distance calculated in self._path_summation(...),
    degeneracy calculated in self._degeneracy_cal.
    Args:
        err_syndrome: Dictionary containing the nodes for subgraph A and B separately
        multi: True for multi-path summation, False ignores such degeneracy
    Returns:
        error_graph: Dictionary containing the error graph ready for MWPM for subgraph A and B separately
    """
    error_graph = [[], []]
    for subgraph in [g.GROUP_A, g.GROUP_B]:
        nodes = err_syndrome[subgraph] + virtual[subgraph] # All the nodes that can be matched
        l_err = len(err_syndrome[subgraph]) # Length of error syndromes
        lo = len(nodes) # Length of all the nodes

        if multi:
            all_pairs_distance_arr = np.zeros((lo, lo), dtype=np.float64)
            all_pairs_degeneracy_arr = np.zeros((lo, lo), dtype=np.uint32)

            for s_ind, t_ind in combinations(range(lo), 2):
                source = nodes[s_ind]
                target = nodes[t_ind]

                if source[0] != 0 or target[0] != 0:
                    sp_gen = nx.all_shortest_paths(S[subgraph], source, target, weight="distance")
                    sp0 = next(sp_gen)
                    distance = 0.0
                    for u_ind in range(len(sp0) - 1):
                        u = sp0[u_ind]
                        v = sp0[u_ind + 1]
                        distance += S[subgraph][u][v]["distance"]
                    degeneracy = 1 + sum(1 for _ in sp_gen)

                    all_pairs_distance_arr[s_ind, t_ind] = distance
                    all_pairs_distance_arr[t_ind, s_ind] = distance
                    all_pairs_degeneracy_arr[s_ind, t_ind] = degeneracy
                    all_pairs_degeneracy_arr[t_ind, s_ind] = degeneracy
        else:
            all_pairs_distance_dict = dict(nx.all_pairs_dijkstra_path_length(S[subgraph], weight="distance"))

        for s_ind, t_ind in combinations(range(lo), 2):
            source = nodes[s_ind]
            target = nodes[t_ind]

            # Distance between virtual nodes is 0
            if source[0] == 0 and target[0] == 0:
                distance = 0.0
            else:
                if multi:
                    distance = all_pairs_distance_arr[s_ind, t_ind]
                    deg = nx_path_summation(
                        lo, l_err, s_ind, t_ind,
                        source[0] == 0, target[0] == 0,
                        all_pairs_distance_arr, all_pairs_degeneracy_arr
                    )
                    distance -= math.log(deg)
                else:
                    distance = float(all_pairs_distance_dict[source][target])

            error_graph[subgraph].append([source, target, distance])

        _append_mwpm_virtual_node(error_graph[subgraph], virtual[subgraph], lo)
        _scale_edge_weights(error_graph[subgraph])

    return error_graph


def error_correct(virtual, matches):
    """
    Error correct according to syndromes, returned values are compared with
    actual logical error to determine the logical error rates.
    Args:
        matches ([(node_a, node_b, edge), ...]): A list of all the matches from MWPM
    Retuns:
        xL (int): The calculated total logical X flips
        zL (int): The calculated total logical Z flips
    """
    xL = zL = 0
    for match in matches:
        for node in match:
            if node in virtual[g.GROUP_A] and node[1] == -0.5:
                zL += 1
            elif node in virtual[g.GROUP_B] and node[2] == -0.5:
                xL += 1

    return xL, zL


def blossom_mwpm_ids(edges):
    """
    Minimum Weight Perfect Matching using node ids (Blossom V algorithm).
    * Node ids are assumed to form a contiguous set of non-negative integers starting at zero, e.g.  {0, 1, ...}.
    * All nodes are assumed to participate in at least one edge.
    :param edges: Edges as [(node_id, node_id, weight), ...].
    :type edges: list of (int, int, int)
    :return: Set of matches as {(node_id, node_id), ...}. (Each tuple is sorted.)
    :rtype: set of (int, int)
    """
    node_ids = sorted(set(id for (id_a, id_b, _) in edges for id in (id_a, id_b)))
    n_nodes = len(node_ids)
    # Check node ids form contiguous set of non-negative integers starting at zero
    assert n_nodes == 0 or (node_ids[0] == 0 and node_ids[-1] == n_nodes - 1), (
        'Node ids are not a contiguous set of non-negative integers starting at zero.')
    n_edges = len(edges)
    nodes_a, nodes_b, weights = zip(*edges) if n_edges else ([], [], [])
    # Prepare arguments
    mates_array_type = ctypes.c_int * n_nodes
    edges_array_type = ctypes.c_int * n_edges
    mates_array = mates_array_type()
    # Call C interface
    _libpypm.mwpm(ctypes.c_int(n_nodes), mates_array, ctypes.c_int(n_edges),
                  edges_array_type(*nodes_a), edges_array_type(*nodes_b), edges_array_type(*weights))
    # Structure of mates: mates_array[i] = j means ith node matches jth node
    # Convert to more useful format: e.g. convert [1, 0, 3, 2] to {(0, 1), (2, 3)}
    mates = {tuple(sorted((a, b))) for a, b in enumerate(mates_array)}
    return mates


def blossom_mwpm(edges):
    """
    Convert to ID-sorted result for MWPM and back.
    :param edges: Edges as [(node, node, weight), ...].
    :type edges: list of (object, object, int)
    :return: Set of matches as {(node, node), ...}.
    :rtype: set of (object, object)
    """
    nodes = list(set(node for (node_a, node_b, _) in edges for node in (node_a, node_b)))
    node_to_id = dict((n, i) for i, n in enumerate(nodes))
    edge_ids = [(node_to_id[node_a], node_to_id[node_b], weight) for node_a, node_b, weight in edges]
    mate_ids = blossom_mwpm_ids(edge_ids)
    mates = {(nodes[node_id_a], nodes[node_id_b]) for node_id_a, node_id_b in mate_ids}

    filtered_mates = [
        (source, target)
        for (source, target) in mates
        if source[0] > 0 or target[0] > 0
    ]

    return filtered_mates
