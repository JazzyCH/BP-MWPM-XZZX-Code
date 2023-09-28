import math
import pandas as pd
import numpy as np

from src.xzzx import XZZXDecoderBase as XZZXDecoder
import src.globals as g

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--shots", type=int, action="store", dest="shots")
parser.add_argument("-d", "--dist", type=int, action="store", dest="dist")
parser.add_argument("-e", "--eta", type=int, action="store", dest="eta")
parser.add_argument("-p", "--prob", type=float, action="store", dest="prob")
parser.add_argument("-c", "--cpu", type=int, action="store", dest="cpu")
parser.add_argument("-p_n", "--prob_n", type=float, action="store", dest="prob_n")


Input = parser.parse_args()
shots = Input.shots
eta = Input.eta
d = Input.dist
p = Input.prob
p_n = Input.prob_n
c = Input.cpu


if __name__ == '__main__':
    fhandle = open("PATH_TO_SAVE_DATA/FILE_NAME.csv", "a")

    if eta == 1:
        N_ROW = N_COL = d
    elif eta == 2:
        N_ROW = N_COL = d
        # N_ROW = d
        # N_COL = math.floor(d * 1.286)
    elif eta == 20:
        N_ROW = d
        # N_COL = math.floor(d * 4.143)
        N_COL = int(round(d * 4.143))
    elif eta == 200:
        N_ROW = d
        # N_COL = math.floor(d * 19.286)
        N_COL = int(round(d * 19.286))
    elif eta == 0:
        N_ROW = 1
        N_COL = d

    if N_COL % 2 == 0:
        N_COL -= 1


    decoder = XZZXDecoder(N_ROW, N_COL, 1, "bp+mwpm", noise_model=[[p, eta, g.ERROR_Z]])
    t_log, x_log, z_log = decoder.run_decoder(shots, 50, num_cpu=c, parallelize=True)


    df = pd.DataFrame([])
    df = pd.DataFrame([[eta, p, d, shots, t_log, x_log, z_log]])
    df.to_csv(fhandle, header=False)
