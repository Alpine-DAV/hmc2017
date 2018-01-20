#!/bin/bash
set -e

APPEND=""
SCHEMA="--schema"
OUTPUT="scaling.csv"

for NCORES in `seq 5 5 50`; do
    srun -A cbronze -N $NCORES -t 1:00:00 -p pbatch --exclusive -o _sbatch.out.hmc_scaling_test.$NCORES.%j.%N.txt \
        python ./scaling.py --output $OUTPUT $APPEND $SCHEMA --verbose \
        /usr/workspace/wsa/hmc_17/data/bubbleShock rf

    APPEND="--append"
    SCHEMA=""
done
