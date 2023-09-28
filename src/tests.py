import numpy as np
from numba import njit

import src.globals as g
import src.bp_legacy as bp_legacy


# Check coordinate<->index conversions
def check_base_coordinate_conversion(decoder, d1, d2):
    print("[DEBUG] JIT coordinate <-> index conversion check")
    l_lst = [
        decoder._get_all_syn_a(),
        decoder._get_all_syn_b(),
        decoder._get_all_syn_real(),
        decoder._get_all_dat()
    ]
    c2i_list = [
        g.syn_a_c2i,
        g.syn_b_c2i,
        g.syn_real_c2i,
        g.dat_c2i
    ]
    i2c_list = [
        g.syn_a_i2c,
        g.syn_b_i2c,
        g.syn_real_i2c,
        g.dat_i2c
    ]

    for t in range(4):
        l = l_lst[t]
        c2i = lambda coord: c2i_list[t](coord, d1, d2)
        i2c = lambda index: i2c_list[t](index, d1, d2)
        print("  - Checking {} and {}".format(
            c2i_list[t].py_func.__name__, i2c_list[t].py_func.__name__
        ))

        print("    - Checking inverses")
        for c in l:
            assert i2c(c2i(c)) == c
        
        for i in range(len(l)):
            assert c2i(i2c(i)) == i

        print("    - Checking bijectivity to array")
        index_list = [c2i(c) for c in l]
        assert sorted(index_list) == list(range(len(l)))


# Sanity check: my left neighbor's right neighbor is me
def check_bp_addr_jit(num_syn, num_dat, addr_array):
    print("[DEBUG] bp.make_addr_table consistency and JIT compatibility check")
    _check_bp_addr_jit_inner(num_syn, num_dat, addr_array)


@njit(cache=True)
def _check_bp_addr_jit_inner(num_syn, num_dat, addr_array):
    for q in range(num_dat):
        q_node = addr_array[g.TANNER_DAT][q]
        for j in q_node[g.TANNER_OCCUPIED][0]:
            inner_ind = q_node[g.TANNER_NB][j][g.TANNER_IND]
            assert q == addr_array[g.TANNER_SYN][inner_ind][g.TANNER_NB][3-j][g.TANNER_IND]

    for p in range(num_syn):
        p_node = addr_array[g.TANNER_SYN][p]
        for i in p_node[g.TANNER_OCCUPIED][0]:
            inner_ind = p_node[g.TANNER_NB][i][g.TANNER_IND]
            assert p == addr_array[g.TANNER_DAT][inner_ind][g.TANNER_NB][3-i][g.TANNER_IND]


# belief_propagation correctness check against legacy implementation
def check_bp_result(rec_array, rounds, syn_nodes, dat_qubits, err_syndrome, err_probs):
    print("[DEBUG] bp.belief_propagate correctness check against legacy implementation")
    tanner_graph = bp_legacy.make_tanner_graph(syn_nodes, dat_qubits, err_syndrome)
    dat_graph = bp_legacy.belief_propagate(syn_nodes, dat_qubits, tanner_graph, err_probs, rounds)

    l = []
    for i in range(2):
        g = tanner_graph[i]
        for k, v in g.items():
            n_arr = []
            n_arr.append(v.get((k[0] - 0.5, k[1])))
            n_arr.append(v.get((k[0], k[1] + 0.5)))
            n_arr.append(v.get((k[0], k[1] - 0.5)))
            n_arr.append(v.get((k[0] + 0.5, k[1])))
            for n in n_arr:
                if n is not None:
                    l.append(n)
    
    assert np.allclose(rec_array, np.array(l))
    return dat_graph


# make_dat_graph correctness check against legacy implementation
def check_bp_dat_graph(dat_graph, dat_graph_legacy, dat_qubits):
    print("[DEBUG] bp.make_dat_graph correctness check against legacy implementation")
    for q in dat_qubits:
        assert q in dat_graph.keys()
        assert q in dat_graph_legacy.keys()
        assert set(dat_graph[q].keys()) == set(dat_graph_legacy[q].keys())
        for k, v in dat_graph[q].items():
            v_l = dat_graph_legacy[q][k]
            assert np.allclose(v, v_l)
