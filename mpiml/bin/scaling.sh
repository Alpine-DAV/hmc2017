#!/bin/bash
set -ev

APPEND=""
SCHEMA="--schema"

for NCORES in 1 `seq 5 5 50`; do
    srun -A cbronze -N $NCORES -t 5:00:00 -p pbatch --exclusive -o _sbatch.out.hmc_scaling_test.$NCORES.%j.%N.txt \
        python ./scaling.py $APPEND $SCHEMA "$@"

    APPEND="--append"
    SCHEMA=""
done
