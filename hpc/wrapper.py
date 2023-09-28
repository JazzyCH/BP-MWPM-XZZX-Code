import os
import numpy as np
from itertools import product

Path = 'FILE_SAVE_PATH'
# dist = np.arange(31,41,2)
# dist = np.arange(3,9,2)
# etas = [1, 2, 20, 200, 2000, 0]
dist = [13, 15, 17]
etas = [1, 2]
# p_ns = [0.025, 0.05, 0.075, 0.1]
shots = 5000
num_cpu = 14

def SendJob(Command, Name):
    Job = Path+'/slurm/'+Name+'.sh'
    f = open(Path+'/slurm/'+Name+'.sh', 'w')
    f.write('#!/bin/bash\n')
    f.write('#SBATCH --output=OUTPUT_PATH/output-%j.txt\n')
    f.write('#SBATCH --mail-type=END\n')
    f.write('#SBATCH --mail-user=NAME@ADDRESS\n')  # Send email notification, change email
    f.write('#SBATCH --partition=day\n')
    f.write('#SBATCH --array=1-20\n')
    f.write('#SBATCH --cpus-per-task=14\n')
    f.write('#SBATCH --job-name=%s\n' % Name)
    f.write('#SBATCH --mem-per-cpu=400MB\n')
    f.write('#SBATCH --time=23:55:00\n\n')
    f.write('%s\n' % Command)
    f.close()
    os.system('sbatch %s' % Job)


if __name__ == '__main__':


    for comb in product(dist, etas):
        d = comb[0]
        eta = comb[1]
        # p_n = comb[2]

        if eta == 1:
            probs = np.linspace(0.044, 0.064, 11)
        elif eta == 2:
            probs = np.linspace(0.082, 0.102, 11)
        elif eta == 20:
             probs = np.linspace(0.24, 0.26, 11)
        elif eta == 200:
            probs = np.linspace(0.375, 0.395, 11)
        else:
            probs = np.linspace(0.4, 0.5, 11)

        for p in probs:
            Command = "python %s/run.py -s %d -d %d -e %d -p %f -c %d" % (Path, shots, d, eta, p, num_cpu)
            Name = "%d_%d_%d_%f_%s" % (shots, d, eta, p, "NAME_OF_CHOICE")
            SendJob(Command, Name)
            print(Command+'\n')
