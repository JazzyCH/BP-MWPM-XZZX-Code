# Global variable declarations
import os
import numpy as np
import awkward as ak
from itertools import product
from numba import njit, float64


DEBUG_FLAG = os.getenv("XZZX_DEBUG")

GROUP_A = 0
GROUP_B = 1

TANNER_SYN = 0
TANNER_DAT = 1
TANNER_ERR = 2 # deprecated, only used in bp_legacy.py for unit tests

TANNER_OCCUPIED = 0
TANNER_NB = 1

TANNER_IND = 0
TANNER_REC = 1

ERROR_I = 0
ERROR_X = 1
ERROR_Y = 2
ERROR_Z = 3

ERRORS = np.array([ERROR_I, ERROR_X, ERROR_Y, ERROR_Z])
ERRORS_CARTESIAN = ak.Array([
    np.array(list(product(ERRORS, repeat=i)))
    for i in range(4)
])


@njit(cache=True)
def syn_a_c2i(coord, d1, d2):
    r, c = int(coord[1] + 0.5), int(coord[2])
    return r * d2 + c


@njit(cache=True)
def syn_a_i2c(index, d1, d2):
    r = index // d2
    c = index % d2
    return float64(0 < r < d1), float64(r - 0.5), float64(c)


@njit(cache=True)
def syn_b_c2i(coord, d1, d2):
    r, c = int(coord[1]), int(coord[2] + 0.5)
    return r * (d2 + 1) + c


@njit(cache=True)
def syn_b_i2c(index, d1, d2):
    r = index // (d2 + 1)
    c = index % (d2 + 1)
    return float64(0 < c < d2), float64(r), float64(c - 0.5)


@njit(cache=True)
def syn_real_c2i(coord, d1, d2):
    r = int(2 * coord[0])
    return int((r // 2) * (2 * d2 - 1) + (r % 2) * (d2 - 0.5) + coord[1] - 0.5)


@njit(cache=True)
def syn_real_i2c(index, d1, d2):
    ex_c = index % (2 * d2 - 1)
    ex_r = (ex_c + 1) // d2
    return float64(index // (2 * d2 - 1) + ex_r * 0.5), float64(ex_c - ex_r * (d2 - 0.5) + 0.5)


@njit(cache=True)
def dat_c2i(coord, d1, d2):
    r = int(2 * coord[0])
    return int((r // 2) * (2 * d2 - 1) + (r % 2) * (d2 - 0.5) + coord[1])


@njit(cache=True)
def dat_i2c(index, d1, d2):
    ex_c = index % (2 * d2 - 1)
    ex_r = ex_c // d2
    return float64(index // (2 * d2 - 1) + ex_r * 0.5), float64(ex_c - ex_r * (d2 - 0.5))
