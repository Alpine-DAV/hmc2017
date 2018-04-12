#!/usr/bin/env bash

set -e

first=true

for num_tasks in `seq 2 8`; do
    if $first; then
        append=""
        schema="--schema"
        first=false
    else
        append="--append"
        schema=""
    fi

    mpiexec -n $num_tasks ./mf_pickle_bench.py $append $schema "$@"
done
