# Profiling
import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.xzzx import XZZXDecoderBase as XZZXDecoder

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: {} <method>, <num_rows>, <num_cols>, <num_shots>".format(sys.argv[0]))
        exit()
    else:
        METHOD = sys.argv[1]
        N_ROW = int(sys.argv[2])
        N_COL = int(sys.argv[3])
        N_SHOTS = int(sys.argv[4])

        # Keywords to choose in decoder type: mwpm, bp, multi
        # noise_model by default is -- [0.04, 1, "Z"] as [biased error rate, bias, bias type]
        decoder = XZZXDecoder(N_ROW, N_COL, 1, METHOD, noise_model=[0.04, 1, "Z"])

        print(decoder.run_decoder(N_SHOTS, 50, num_cpu=1, parallelize=False, display_progress=True))
