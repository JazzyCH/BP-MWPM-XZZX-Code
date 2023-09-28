import numpy as np
import copy
import math
from itertools import combinations, product
import matplotlib.pyplot as plt
from profilehooks import profile, coverage, timecall
from numba import njit
import awkward as ak

import src.globals as g


def make_addr_table(d1, d2, syn_nodes, dat_qubits):
    rec_ptr = 0
    addr = [
        [[[[]], [None, None, None, None]] for i in range(len(syn_nodes))], # syndrome, [occupied, nb]
        [[[[]], [None, None, None, None]] for i in range(len(dat_qubits))] # data, [occupied, nb]
    ]
    init_records = []

    for coord in syn_nodes:
        ind = g.syn_real_c2i(coord, d1, d2)

        neighbors = [
            (coord[0] - 0.5, coord[1]),
            (coord[0], coord[1] + 0.5),
            (coord[0], coord[1] - 0.5),
            (coord[0] + 0.5, coord[1])
        ]

        syn_ind = addr[g.TANNER_SYN][ind]
        for i in range(4):
            if neighbors[i] in dat_qubits:
                syn_ind[g.TANNER_OCCUPIED][0].append(i)
                syn_ind[g.TANNER_NB][i] = [
                    g.dat_c2i(neighbors[i], d1, d2),
                    rec_ptr
                ]
                init_records.append([1.0, 1.0, 1.0, 1.0])
                rec_ptr += 1

    num_syn_rec = rec_ptr

    for coord in dat_qubits:
        ind = g.dat_c2i(coord, d1, d2)

        neighbors = [
            (coord[0] - 0.5, coord[1]),
            (coord[0], coord[1] + 0.5),
            (coord[0], coord[1] - 0.5),
            (coord[0] + 0.5, coord[1])
        ]

        dat_ind = addr[g.TANNER_DAT][ind]
        for i in range(4):
            if neighbors[i] in syn_nodes:
                dat_ind[g.TANNER_OCCUPIED][0].append(i)
                dat_ind[g.TANNER_NB][i] = [
                    g.syn_real_c2i(neighbors[i], d1, d2),
                    rec_ptr
                ]
                init_records.append([-1.0, -1.0, -1.0, -1.0])
                rec_ptr += 1

        num_dat_rec = rec_ptr - num_syn_rec

    return ak.Array(addr), np.array(init_records, dtype=np.float64), (rec_ptr, num_syn_rec, num_dat_rec)


def make_error_array(d1, d2, syn_nodes, err_syndrome):
    err_array = np.zeros(len(syn_nodes), dtype=np.bool_)
    for coord in err_syndrome:
        ind = g.syn_real_c2i(coord, d1, d2)
        err_array[ind] = 1
    return err_array


@njit
def qubit_to_check(q, addr_array, rec_array, err_probs, last_check=False):
    # Load info for making combinations of neighboring checks
    q_node = addr_array[g.TANNER_DAT][q]
    occupied = q_node[g.TANNER_OCCUPIED][0]
    nb = q_node[g.TANNER_NB]

    # Boundary condition for d X 1 lattice
    if len(occupied) == 1:
        nb_ind = occupied[0]
        lat_ind, rec_ind = nb[nb_ind]
        if np.all(rec_array[rec_ind] >= 0):
            nb_rec_ind = addr_array[g.TANNER_SYN][lat_ind][g.TANNER_NB][3 - nb_ind][g.TANNER_REC]
            rec_array[rec_ind] = rec_array[nb_rec_ind]
            if last_check:
                rec_array[rec_ind] *= err_probs
                rec_array[rec_ind] /= np.sum(rec_array[rec_ind])
        else:
            rec_array[rec_ind] = err_probs

    else:
        for n1 in occupied:
            rec_ind = nb[n1][g.TANNER_REC]
            rec_array[rec_ind] = err_probs
            for n2 in occupied:
                if n2 != n1:
                    n2_ind = nb[n2][g.TANNER_IND]
                    q_rec = addr_array[g.TANNER_SYN][n2_ind][g.TANNER_NB][3 - n2][g.TANNER_REC]
                    rec_array[rec_ind] *= rec_array[q_rec]
            rec_array[rec_ind] /= np.sum(rec_array[rec_ind])


@njit
def check_to_qubit(d1, d2, c, addr_array, rec_array, is_error):
    c_node = addr_array[g.TANNER_SYN][c]
    occupied = c_node[g.TANNER_OCCUPIED][0]
    lo = len(occupied)
    nb = c_node[g.TANNER_NB]
    c_r, c_c = g.syn_real_i2c(c, d1, d2)

    rev_rec_arr = np.empty((lo, 4), dtype=np.float64)
    n_coord_arr = np.empty((lo, 2), dtype=np.float64)

    for i in range(lo):
        n_ind = nb[occupied[i]][g.TANNER_IND]
        n_node = addr_array[g.TANNER_DAT][n_ind]
        rev_rec_arr[i] = rec_array[n_node[g.TANNER_NB][3 - occupied[i]][g.TANNER_REC]]
        n_coord_arr[i] = g.dat_i2c(n_ind, d1, d2)

    # iteration for n1
    for i in range(lo):
        n1 = occupied[i]
        rec_ind = nb[n1][g.TANNER_REC]
        n1_r, n1_c = n_coord_arr[i]

        rem = np.array([j for j in range(lo) if j != i])

        msg = np.empty(4, dtype=np.float64)

        for fixed_err in g.ERRORS:
            flip = 0
            if fixed_err == g.ERROR_Y:
                flip += 1
            elif fixed_err == g.ERROR_X and n1_r != c_r:
                flip += 1
            elif fixed_err == g.ERROR_Z and n1_c != c_c:
                flip += 1

            to_flip = flip + is_error

            f_msg = 0.0
            for marg in g.ERRORS_CARTESIAN[lo - 1]:
                flip = 0
                for i in range(lo - 1):
                    if marg[i] == g.ERROR_Y:
                        flip += 1
                    elif marg[i] == g.ERROR_X and n_coord_arr[rem[i]][0] != c_r:
                        flip += 1
                    elif marg[i] == g.ERROR_Z and n_coord_arr[rem[i]][1] != c_c:
                        flip += 1
                if flip % 2 == to_flip % 2:
                    s_msg = 1.0
                    for i in range(lo - 1):
                        s_msg *= rev_rec_arr[rem[i]][marg[i]]
                else:
                    s_msg = 0.0

                f_msg += s_msg

            msg[fixed_err] = f_msg

        msg /= np.sum(msg)
        rec_array[rec_ind] = msg


@njit
def belief_propagate(
        d1, d2,
        num_syn, num_dat, addr_array, rec_array, err_array,
        err_probs, rounds, plot_cvg=False
    ):

    # if plot_cvg:
    #     rec_hist = np.empty((rounds + 1, len(rec_array), len(rec_array[0])), dtype=np.float64)
    #     rec_hist[0] = rec_array

    for r in range(rounds):
        if r != rounds - 1:
            for q in range(num_dat):
                qubit_to_check(q, addr_array, rec_array, err_probs, last_check=False)
            for c in range(num_syn):
                check_to_qubit(d1, d2, c, addr_array, rec_array, err_array[c])
        else:
            for q in range(num_dat):
                qubit_to_check(q, addr_array, rec_array, err_probs, last_check=True)

        # if plot_cvg:
        #     rec_hist[r + 1] = rec_array

    # if plot_cvg:
    #     errors = ["I", "X", "Y", "Z"]
    #     for pos in dat_qubits:
    #         figsize=(6, 6)
    #         fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, tight_layout=True)
    #         index = 0
    #         for ax in axes.ravel():
    #             record = np.arange(1, rounds+1)
    #             for neighbor in check_graph[pos]:
    #                 data = (check_graph[pos][neighbor]).T
    #                 data = data[index][1:]
    #                 ax.plot(record, data)
    #                 ax.set_title("%s probability" % (errors[index]))

    #             index += 1
    #         fig.suptitle("qubit position: (%.1f, %.1f)" % (pos[0], pos[1]), y=1.05)
    #         fig.tight_layout()
    #         plt.show()


# This method is just a patchwork to add compatibility with legacy code below
# TODO: Rewrite data structures to get rid of this
def make_dat_graph(syn_nodes, dat_qubits, rec_array, num_syn_rec):
    ind = num_syn_rec
    dat_graph = {}
    for coord in dat_qubits:
        neighbors = [
            (coord[0] - 0.5, coord[1]),
            (coord[0], coord[1] + 0.5),
            (coord[0], coord[1] - 0.5),
            (coord[0] + 0.5, coord[1])
        ]

        dat_graph[coord] = {}
        for n in neighbors:
            if n in syn_nodes:
                dat_graph[coord][n] = rec_array[ind]
                ind += 1

    return dat_graph


# Use if using activation function
def process_dat_graph(dat_qubits, dat_graph):
    new_graph = {pos: {g.ERROR_Z: None, g.ERROR_X: None, "err": None} for pos in dat_qubits}

    for pos in dat_graph:
        arr = []
        for neighbor in dat_graph[pos]:
            if np.shape(arr)[0] < 1:
                arr = dat_graph[pos][neighbor]
            else:
                arr = np.vstack([arr, dat_graph[pos][neighbor]])
        arr = arr.T
        # Edge case: d X 1 lattice
        try:
            result = np.mean(arr, axis=1)
        except:
            result = arr
        cut = np.where(result == np.max(result))[0]
        if len(cut) != 1:
            raise ValueError("Maximum probability shared among multiple error types.")

        error = g.ERRORS[cut]
        new_graph[pos]["err"] = error
        pi = result[0]
        px = result[1]
        py = result[2]
        pz = result[3]

        if not (pz == 0 and py == 0):
            weight = math.log(pi / (pz + py))
            weight = 10 / (1+ math.exp(- 0.5*(weight - 5)))
            new_graph[pos][g.ERROR_Z] = weight

        if not (px == 0 and py == 0):
            weight = math.log(pi / (px + py))
            weight = 10 / (1+ math.exp(- 0.5*(weight - 5)))
            new_graph[pos][g.ERROR_X] = weight

    return new_graph

# Use if not using activation functions
def lattice_dat_graph(dat_qubits, dat_graph):
    new_graph = {pos: {g.ERROR_Z: None, g.ERROR_X: None, "err": None} for pos in dat_qubits}
    min_pos = 10000

    for pos in dat_graph:
        arr = []
        for neighbor in dat_graph[pos]:
            if np.shape(arr)[0] < 1:
                arr = dat_graph[pos][neighbor]
            else:
                arr = np.vstack([arr, dat_graph[pos][neighbor]])
        arr = arr.T
        # Edge case: d X 1 lattice
        try:
            result = np.mean(arr, axis=1)
        except:
            result = arr
        cut = np.where(result == np.max(result))[0]
        if len(cut) != 1:
            raise ValueError("Maximum probability shared among multiple error types.")

        error = g.ERRORS[cut]
        new_graph[pos]["err"] = error
        pi = result[0]
        px = result[1]
        py = result[2]
        pz = result[3]

        if not (pz == 0 and py == 0):
            weight = math.log(pi / (pz + py))
            if weight > 0 and weight < min_pos:
                min_pos = weight

            new_graph[pos][g.ERROR_Z] = weight

        if not (px == 0 and py == 0):
            weight = math.log(pi / (px + py))
            if weight > 0 and weight < min_pos:
                min_pos = weight

            new_graph[pos][g.ERROR_X] = weight

    # # Force the constraint on nonnegative edge weights
    # for pos in new_graph:
    #     if new_graph[pos][g.ERROR_Z] != None and new_graph[pos][g.ERROR_Z] < 0:
    #         if new_graph[pos]["err"] == g.ERROR_X:
    #             new_graph[pos][g.ERROR_Z] = min_pos
    #         else:
    #             new_graph[pos][g.ERROR_Z] = 0

    #     if new_graph[pos][g.ERROR_X] != None and new_graph[pos][g.ERROR_X] < 0:
    #         if new_graph[pos]["err"] == g.ERROR_Z:
    #             new_graph[pos][g.ERROR_X] = min_pos
    #         else:
    #             new_graph[pos][g.ERROR_X] = 0

    return new_graph



def error_correct(dat_graph, rows, cols):

    # Consider logical qubits only to improve efficiency
    xL_qubits = [(r, 0) for r in range(rows)]
    zL_qubits = [(0, c) for c in range(cols)]
    log_qubits = list(set(xL_qubits + zL_qubits))

    xL = zL = 0

    for pos in log_qubits:
        arr = []
        for neighbor in dat_graph[pos]:
            if np.shape(arr)[0] < 1:
                arr = dat_graph[pos][neighbor]
            else:
                arr = np.vstack([arr, dat_graph[pos][neighbor]])
        arr = arr.T
        # Edge case: d X 1 lattice
        try:
            result = np.mean(arr, axis=1)
        except:
            result = arr
        cut = np.where(result == np.max(result))[0]
        if len(cut) != 1:
            raise ValueError("Maximum probability shared among multiple error types.")

        error = g.ERRORS[cut]
        if pos[0] == 0 and (error == g.ERROR_X or error == g.ERROR_Y):
            zL += 1
        if pos[1] == 0 and (error == g.ERROR_Z or error == g.ERROR_Y):
            xL += 1

    return xL, zL
