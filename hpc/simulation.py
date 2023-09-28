import math
import pandas as pd
import numpy as np
from src.xzzx import XZZXDecoderBase as XZZXDecoder

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--shots", type=int, action="store", dest="shots")
parser.add_argument("-d", "--dist", type=int, action="store", dest="dist")
parser.add_argument("-e", "--eta", type=int, action="store", dest="eta")
parser.add_argument("-p", "--prob", type=float, action="store", dest="prob")
parser.add_argument("-c", "--cpu", type=int, action="store", dest="cpu")

Input = parser.parse_args()
shots = Input.shots 
eta = Input.eta
d = Input.dist
p = Input.prob
c = Input.cpu


if __name__ == '__main__':
    fhandle = open("PATH_TO_FILE/FILE_NAME", "a")


    if eta == 1:
        N_ROW = N_COL = d
    elif eta == 0:
        N_ROW = 1
        N_COL = d
    else:
        N_ROW = d
        p_total = p + p * 2 / eta
        N_COL = d*(1 - math.log(eta/2) / math.log(p_total))
        N_COL = math.floor(N_COL)

        if N_COL % 2 == 0:
            N_COL -= 1

    decoder = XZZXDecoder(N_ROW, N_COL, 1, "bp+mwpm", noise_model=[p, eta, "Z"])
    t_log, x_log, z_log = decoder.run_decoder(shots, 50, num_cpu=c, parallelize=True)


    df = pd.DataFrame([])
    df = pd.DataFrame([[eta, p, d, shots, t_log, x_log, z_log]])
    df.to_csv(fhandle, header=False)
