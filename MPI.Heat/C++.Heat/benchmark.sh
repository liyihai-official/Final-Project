#!/bin/bash
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 24:00:00
#SBATCH -J First_Benchmark_Final_Project
#SBATCH -o benchmark_first.o.%j
#SBATCH -e benchmark_first.e.%j
#SBATCH --mail-user=liy35@tcd.ie
#SBATCH --mail-type=ALL
#SBATCH --account=liy35

# Load Enssential Module
module load openmpi/4.1.6-gcc-13.1.0-kzjsbji

# File: benchmark.sh
# Create the benchmark.results directory if it doesn't exist
if [ ! -d "benchmark.results" ]; then
    mkdir benchmark.results
fi

# Set the base dimensions
BASE_N_X=16
BASE_N_Y=16

# Define the range of processor counts for scaling tests
PROCS_WEAK=(1 2 3 4 5 6 7 8)
PROCS_STRONG=(1 2 3 4 5 6 7 8)

# Strong Scaling Test
echo "Strong Scaling"
for PROC in "${PROCS_STRONG[@]}"; do
    echo "Running with $((PROC*PROC)) processes \t Grid size ${SCALE_N_X}x${SCALE_N_Y}"
    make clean
    make MAX_N_X=-DMAX_N_X=${SCALE_N_X} MAX_N_Y=-DMAX_N_Y=${SCALE_N_Y}
    mpirun -np $((PROC*PROC)) ./main > benchmark.results/strong_main_${PROC}_${SCALE_N_X}_${SCALE_N_Y}
done

echo "\n"

# Weak Scaling Test
echo "Weak Scaling"
for PROC in "${PROCS_WEAK[@]}"; do
    SCALE_N_X=$((BASE_N_X * PROC))
    SCALE_N_Y=$((BASE_N_Y * PROC))
    echo "Running with $((PROC*PROC)) processes \t Grid size ${SCALE_N_X}x${SCALE_N_Y}"
    make clean
    make MAX_N_X=-DMAX_N_X=${SCALE_N_X} MAX_N_Y=-DMAX_N_Y=${SCALE_N_Y}
    mpirun -np $((PROC*PROC)) ./main > benchmark.results/weak_main_${PROC}_${SCALE_N_X}_${SCALE_N_Y}
done
