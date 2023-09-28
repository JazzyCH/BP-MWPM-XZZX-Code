from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import awkward as ak
import networkx as nx
import os

from src.noise import *
from src.utils import *
import src.bp as bp
import src.mwpm as mwpm
import src.globals as g
import src.tests as t


if g.DEBUG_FLAG:
    print("=== XZZX DECODER: DEBUG TESTS ACTIVE ===")


class XZZXDecoderBase:
    """
    Class to construct XZZX code on a rotated lattice.
    """
    def __init__(self, d1, d2, T, decoder, noise_model = [0.04, 1, g.ERROR_Z]):
        # Currently consider T = 1 only for the hashing bound
        # And biased noise channel only
        self.__row = d1
        self.__col = d2
        self.__T = T
        self.__decoder = decoder # Specify the decoder type, keywords: bp, mwpm, bp+mwpm
        self.__err_rate, self.__bias_eta, self.__bias_type = noise_model
        self.__err_probs = biased_hashing(*noise_model)
        self.__is_depolarized = (noise_model[1] == 1)

        pi, px, py, pz = self.__err_probs
        self.__z_edge_weight = math.log(pi / (pz + py))
        self.__x_edge_weight = math.log(pi / (px + py))

        if "multi" in self.__decoder:
            self.__m = True
        else:
            self.__m = False

        if not ("bp" in self.__decoder and "mwpm" not in self.__decoder):
            self.__virtual = self._specify_virtual()
        self.__syn_nodes = self._specify_syndrome()
        if "bp" in self.__decoder:
            self.__all_nodes = [
                node[1:] for node in self.__syn_nodes[g.GROUP_A]
            ] + [
                node[1:] for node in self.__syn_nodes[g.GROUP_B]
            ]
            self.__num_syn = len(self.__all_nodes)
        self.__dat_qubits = self._specify_dat()
        self.__num_dat = len(self.__dat_qubits)

        if g.DEBUG_FLAG:
            t.check_base_coordinate_conversion(self, self.__row, self.__col)

        if "bp" in self.__decoder:
            self.__bp_addr, self.__bp_init_rec, num_rec = bp.make_addr_table(
                self.__row, self.__col, self.__all_nodes, self.__dat_qubits
            )
            self.__bp_num_rec, self.__bp_num_syn_rec, self.__bp_num_dat_rec = num_rec

            if g.DEBUG_FLAG:
                t.check_bp_addr_jit(self.__num_syn, self.__num_dat, self.__bp_addr)


    def _get_all_syn_a(self):
        return self.__syn_nodes[g.GROUP_A] + self.__virtual[g.GROUP_A]


    def _get_all_syn_b(self):
        return self.__syn_nodes[g.GROUP_B] + self.__virtual[g.GROUP_B]


    def _get_all_syn_real(self):
        return self.__all_nodes


    def _get_all_dat(self):
        return self.__dat_qubits


    def _specify_virtual(self):
        """
        Returns the virtual syndrome nodes for subgraph A and B separately in dictionary
        Returns:
            virtual = [[(0.0, r, c), ... ], [(0.0, r, c), ...]]
        """
        virtual = [[], []]
        for i in range(self.__col):
            virtual[g.GROUP_A].append((0.0, -0.5, float(i)))
            virtual[g.GROUP_A].append((0.0, self.__row - 0.5, float(i)))
        virtual[g.GROUP_A] = list(set(virtual[g.GROUP_A]))

        for i in range(self.__row):
            virtual[g.GROUP_B].append((0.0, float(i), -0.5))
            virtual[g.GROUP_B].append((0.0, float(i), self.__col - 0.5))
        virtual[g.GROUP_B] = list(set(virtual[g.GROUP_B]))

        return virtual


    def _specify_syndrome(self):
        nodes = [[], []]

        for t in range(self.__T):
            nodes[g.GROUP_A] += [
                (float(t + 1), r + 0.5, float(c))
                for r in range(self.__row - 1)
                for c in range(self.__col)
            ]

            nodes[g.GROUP_B] += [
                (float(t + 1), float(r), c + 0.5)
                for c in range(self.__col - 1)
                for r in range(self.__row)
            ]

        return nodes


    def _specify_dat(self):
        """
        Returns the data qubits for the whole graph in a list
        Returns:
            data_qubits = [(r, c), ...]
        """
        dat_qubits = [
            (float(q[0]), float(q[1]))
            for q in product(range(self.__row), range(self.__col))
        ]
        for r in range(self.__row - 1):
            for c in range(self.__col - 1):
                dat_qubits.append((r + 0.5, c + 0.5))

        return dat_qubits


    def _make_syndrome_graph(self, noise_graph):
        # TODO fix/test the edge weight problem
        S = [nx.Graph(), nx.Graph()]
        for subgraph in [g.GROUP_A, g.GROUP_B]:
            for curr_node in self.__syn_nodes[subgraph]:

                if not S[subgraph].has_node(curr_node):
                    S[subgraph].add_node(curr_node)

                r, c = curr_node[1], curr_node[2]
                neighbors = [
                    (r - 1, c, g.ERROR_X, (r - 0.5, c)),
                    (r + 1, c, g.ERROR_X, (r + 0.5, c)),
                    (r, c - 1, g.ERROR_Z, (r, c - 0.5)),
                    (r, c + 1, g.ERROR_Z, (r, c + 0.5))
                ]

                for target in neighbors:
                    if self._valid_syndrome(target, subgraph):
                        target_node = (curr_node[0],) + target[:-2]
                    elif self._valid_virtual(target, subgraph):
                        target_node = (0,) + target[:-2]
                    else:
                        continue

                    if not S[subgraph].has_node(target_node):
                        S[subgraph].add_node(target_node)

                    weight = noise_graph[target[3]][target[2]]
                    if weight == None:
                        continue

                    # S[subgraph].add_edge(
                    #     curr_node, target_node, distance=weight
                    # )

                    if weight < 0:
                        S[subgraph].add_edge(
                            curr_node, target_node, distance=weight, color='r'
                        )
                    else:
                        S[subgraph].add_edge(
                            curr_node, target_node, distance=weight, color='b'
                        )

        return S


    def _valid_syndrome(self, node, subgraph):
        r, c = node[0], node[1]

        if subgraph == g.GROUP_A:
            if (
                r > 0
                and r < self.__row - 1
                and c > -0.5
                and c < self.__col - 0.5
            ):
                return True
            else:
                return False
        else:
            if (
                r > -0.5
                and r < self.__row - 0.5
                and c > 0
                and c < self.__col - 1
            ):
                return True
            else:
                return False


    def _valid_virtual(self, node, subgraph):
        r, c = node[0], node[1]

        if subgraph == g.GROUP_A and c > -0.5 and c < self.__col - 0.5:
            return True
        elif subgraph == g.GROUP_B and r > -0.5 and r < self.__row - 0.5:
            return True
        else:
            return False


    def _run_bp_mwpm(self, rounds, bp_check=False, lattice_check=False):
        # Can run BP alone also
        err_syndrome, xL, zL, flipped = get_error_syndrome(
            self.__syn_nodes, self.__dat_qubits, self.__err_probs
        )

        # err_syndrome_graph(self.__syn_nodes, self.__virtual, err_syndrome, flipped)
        bp_err = err_syndrome[g.GROUP_A] + err_syndrome[g.GROUP_B]
        for i in range(len(bp_err)):
            bp_err[i] = bp_err[i][1:]

        err_array = bp.make_error_array(self.__row, self.__col, self.__all_nodes, bp_err)
        rec_array = self.__bp_init_rec.copy()

        bp.belief_propagate(
            self.__row, self.__col,
            self.__num_syn, self.__num_dat, self.__bp_addr, rec_array, err_array,
            self.__err_probs, rounds, plot_cvg=bp_check
        )

        dat_graph = bp.make_dat_graph(
            self.__all_nodes, self.__dat_qubits, rec_array, self.__bp_num_syn_rec
        )

        if g.DEBUG_FLAG:
            dat_graph_legacy = t.check_bp_result(
                rec_array, rounds,
                self.__all_nodes, self.__dat_qubits, bp_err, self.__err_probs
            )
            t.check_bp_dat_graph(
                dat_graph, dat_graph_legacy, self.__dat_qubits
            )

        if "mwpm" in self.__decoder:
            if lattice_check:
                noise_graph = bp.lattice_dat_graph(self.__dat_qubits, dat_graph)
                S = self._make_syndrome_graph(noise_graph)
                nx_2D_lattice_color(S[g.GROUP_A], "distance", flipped)
                nx_2D_lattice_color(S[g.GROUP_B], "distance", flipped)
                return 0, 0

            noise_graph = bp.process_dat_graph(self.__dat_qubits, dat_graph)
            S = self._make_syndrome_graph(noise_graph)

            error_graph = mwpm.nx_error_graph(S, self.__virtual, err_syndrome, multi=self.__m)
            matches_a = mwpm.blossom_mwpm(error_graph[g.GROUP_A])
            matches_b = mwpm.blossom_mwpm(error_graph[g.GROUP_B])
            xL_a, zL_a = mwpm.error_correct(self.__virtual, matches_a)
            xL_b, zL_b = mwpm.error_correct(self.__virtual, matches_b)
            xL_t = xL_a + xL_b
            zL_t = zL_a + zL_b
        else:
            xL_t, zL_t = bp.error_correct(dat_graph, self.__row, self.__col)

        return int(xL_t % 2 != xL % 2), int(zL_t % 2 != zL % 2)


    def _run_mwpm(self):
        # This by default uses the blossom VI mwpm structure for better performance
        err_syndrome, xL, zL, flipped = get_error_syndrome(
            self.__syn_nodes, self.__dat_qubits, self.__err_probs
        )
        # err_syndrome_graph(self.__syn_nodes, self.__virtual, err_syndrome, flipped)
        error_graph = mwpm.blossom_error_graph(
            self.__virtual, err_syndrome, self.__err_probs,
            self.__z_edge_weight, self.__x_edge_weight,
            self.__is_depolarized, multi=self.__m
        )
        matches_a = mwpm.blossom_mwpm(error_graph[g.GROUP_A])
        matches_b = mwpm.blossom_mwpm(error_graph[g.GROUP_B])
        xL_a, zL_a = mwpm.error_correct(self.__virtual, matches_a)
        xL_b, zL_b = mwpm.error_correct(self.__virtual, matches_b)
        xL_t = xL_a + xL_b
        zL_t = zL_a + zL_b

        return int(xL_t % 2 != xL % 2), int(zL_t % 2 != zL % 2)


    def run_decoder(self,
        shots, rounds=0,
        bp_graph=False, lattice_graph=False, display_progress=True,
        num_cpu=None, parallelize=True
    ):
        if num_cpu == None:
            num_cpu = os.cpu_count()

        # Sanity checks
        if num_cpu < 1:
            raise ValueError("Please enter a valid number of CPUs.")
        if "bp" in self.__decoder and rounds == 0:
            raise ValueError("Please enter a valid round number for belief propagation.")
        if "bp" in self.__decoder and (self.__row == 1 or self.__col == 1) and self.__bias_eta != 0:
            raise NotImplementedError("Only pure noise models are supported on 1-dimensional lattices.")

        if lattice_graph:
            base_2D_lattice(self.__syn_nodes, self.__virtual)

        if parallelize:
            # Setup per-shot decoder
            if "bp" in self.__decoder:
                run_decoder_shot = delayed(self._run_bp_mwpm)(rounds, bp_graph, lattice_graph)
            else:
                run_decoder_shot = delayed(self._run_mwpm)()

            with tqdm_joblib(tqdm(
                    desc="\033[1m[MP-{}]\033[0m {}, d={}x{}, T={}, P={}"
                    .format(
                        num_cpu, self.__decoder, self.__row, self.__col, self.__T, self.__err_probs
                    ),
                    total=shots, disable=not display_progress
                )) as _:
                res = Parallel(
                        n_jobs=num_cpu
                    )(run_decoder_shot for s in range(shots))
        else:
            if "bp" in self.__decoder:
                res = [
                    self._run_bp_mwpm(rounds, bp_graph, lattice_graph)
                    for s in tqdm(
                        range(shots),
                        desc="\033[1m[SP]\033[0m {}, d={}x{}, T={}, P={}"
                        .format(
                            self.__decoder, self.__row, self.__col, self.__T, self.__err_probs
                        ),
                        disable=not display_progress
                    )
                ]
            else:
                res = [
                    self._run_mwpm()
                    for s in tqdm(
                        range(shots),
                        desc="\033[1m[SP]\033[0m {}, d={}x{}, T={}, P={}"
                        .format(
                            self.__decoder, self.__row, self.__col, self.__T, self.__err_probs
                        ),
                        disable=not display_progress
                    )
                ]

        # Add t_log
        res = np.array(res).T
        res = np.insert(res, 0, np.logical_or(res[0], res[1]), axis=0)

        return np.sum(res, axis=1) / shots
