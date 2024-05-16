#!/bin/bash

# File: benchmark.sh
# Create the benchmark.results directory if it doesn't exist
if [ ! -d "benchmark.results" ]; then
    mkdir benchmark.results
fi

# Set the base dimensions
BASE_N_X=32
BASE_N_Y=32

# Define the range of processor counts for scaling tests
PROCS_WEAK=(1 2 4 8)
PROCS_STRONG=(1 2 4 8)

# Weak Scaling Test
echo "Weak Scaling"
for PROC in "${PROCS_WEAK[@]}"; do
    SCALE_N_X=$((BASE_N_X * PROC))
    SCALE_N_Y=$((BASE_N_Y * PROC))
    echo "Running with ${PROC} processes and grid size ${SCALE_N_X}x${SCALE_N_Y}"
    make clean
    make MAX_N_X=-DMAX_N_X=${SCALE_N_X} MAX_N_Y=-DMAX_N_Y=${SCALE_N_Y}
    mpirun -np ${PROC} ./pro_main > benchmark.results/weak_pro_main_${PROC}_${SCALE_N_X}_${SCALE_N_Y}
done

# Strong Scaling Test
echo "Strong Scaling"
for PROC in "${PROCS_STRONG[@]}"; do
    echo "Running with ${PROC} processes and grid size ${BASE_N_X}x${BASE_N_Y}"
    make clean
    make MAX_N_X=-DMAX_N_X=${BASE_N_X} MAX_N_Y=-DMAX_N_Y=${BASE_N_Y}
    mpirun -np ${PROC} ./pro_main > benchmark.results/strong_pro_main_${PROC}_${SCALE_N_X}_${SCALE_N_Y}
done
