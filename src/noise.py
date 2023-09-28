import math
from random import choices
import numpy as np
import src.globals as g


def biased_hashing(p, eta, bias="Z"):
    # eta <= 0 is infinite bias
    if bias == "Z":
        pz = p
        if eta <= 0: px = py = 0
        else: px = py = p / eta
    elif bias == "X":
        px = p
        if eta <= 0: pz = py = 0
        else: pz = py = p / eta
    elif bias == "Y":
        py = p
        if eta <= 0: px = pz = 0
        else: px = pz = p / eta
    else:
        raise ValueError("The error label should be X, Y, or Z")

    if px + py + pz > 1:
        raise ValueError("The error probability is too high")

    return np.array([1 - px - py - pz, px, py, pz])


def get_error_syndrome(S, dat_qubits, err_probs):
    """
    Populate error on each data qubits, propagate errors to neighboring syndomes.
    Returns the actual logical error and the syndrome flips for error correction.
    Returns:
        err_syndrome: Dictionary containing the nodes for subgraph A and B separately
        xL (int): The actual total logical X flips
        zL (int): The actual total logical Z flips
    """
    # IMPORTANT: we are now assuming only initialization error on the qubits
    anc_error = [{pos: 0 for pos in S[g.GROUP_A]}, {pos: 0 for pos in S[g.GROUP_B]}]

    err_syndrome = [{}, {}]
    xL = zL = 0

    flipped = [None, [], [], []] # I, X, Y, Z
    for pos in dat_qubits:
        # Determine error or not according to error probability
        error = choices([g.ERROR_I, g.ERROR_X, g.ERROR_Y, g.ERROR_Z], weights = err_probs)[0]

        if error == g.ERROR_X:
            flipped[g.ERROR_X].append(pos)
        if error == g.ERROR_Y:
            flipped[g.ERROR_Y].append(pos)
        if error == g.ERROR_Z:
            flipped[g.ERROR_Z].append(pos)

        # Count actual logical error from data qubits
        if pos[0] == 0 and (error == g.ERROR_X or error == g.ERROR_Y):
            zL += 1
        if pos[1] == 0 and (error == g.ERROR_Z or error == g.ERROR_Y):
            xL += 1

        up = (1, pos[0]-0.5, pos[1])
        down = (1, pos[0]+0.5, pos[1])
        left = (1, pos[0], pos[1]-0.5)
        right = (1, pos[0], pos[1]+0.5)

        # Propagate error from qubit to ancilla
        if up in S[g.GROUP_A] and (error == g.ERROR_X or error == g.ERROR_Y):
            anc_error[g.GROUP_A][up] += 1
        elif up in S[g.GROUP_B] and (error == g.ERROR_X or error == g.ERROR_Y):
            anc_error[g.GROUP_B][up] += 1

        if down in S[g.GROUP_A] and (error == g.ERROR_X or error == g.ERROR_Y):
            anc_error[g.GROUP_A][down] += 1
        elif down in S[g.GROUP_B] and (error == g.ERROR_X or error == g.ERROR_Y):
            anc_error[g.GROUP_B][down] += 1

        if left in S[g.GROUP_A] and (error == g.ERROR_Z or error == g.ERROR_Y):
            anc_error[g.GROUP_A][left] += 1
        elif left in S[g.GROUP_B] and (error == g.ERROR_Z or error == g.ERROR_Y):
            anc_error[g.GROUP_B][left] += 1

        if right in S[g.GROUP_A] and (error == g.ERROR_Z or error == g.ERROR_Y):
            anc_error[g.GROUP_A][right] += 1

        elif right in S[g.GROUP_B] and (error == g.ERROR_Z or error == g.ERROR_Y):
            anc_error[g.GROUP_B][right] += 1

    # Determine error syndromes in subgraph A and B
    for subgraph in [g.GROUP_A, g.GROUP_B]:
        err_syndrome[subgraph] = [
            pos
            for pos in anc_error[subgraph]
            if anc_error[subgraph][pos] % 2 != 0
        ]

    return err_syndrome, xL, zL, flipped
