#!/bin/bash

METHOD=$1
N_ROW=$2
N_COL=$3
N_SHOTS=$4
LABEL=$5

PROFILE_NAME=$(git rev-parse --short HEAD)
PROFILE_DIR="profiling"

FN=${PROFILE_DIR}/${PROFILE_NAME}_${METHOD}_${N_ROW}_${N_COL}_${N_SHOTS}
[ ! -z ${LABEL} ] && FN=${FN}_${LABEL}

python -m cProfile -o ${FN}.pstats profile-driver.py ${METHOD} ${N_ROW} ${N_COL} ${N_SHOTS}
gprof2dot -f pstats ${FN}.pstats | dot -Tpdf -o ${FN}.pdf

printf "\n\nProfiling results saved to ./${FN}.pdf.\n\n"
