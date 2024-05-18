#!/bin/bash
#SBATCH -N 4
#SBATCH -n 256
#SBATCH -t 24:00:00
#SBATCH -J scalingtest
#SBATCH -o benchmark_first.o.%j
#SBATCH -e benchmark_first.e.%j
#SBATCH --mail-user=liy35@tcd.ie
#SBATCH --mail-type=ALL
#SBATCH --account=callan_liy35
#SBATCH --partition=compute

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
SCALE_=(0 1)

# Weak Scaling Test
echo "Weak Scaling"
for SCALE in "${SCALE_[@]}"; do
    PROC=$((2 ** (2*SCALE)))
    SIZE_N_X=$((2 ** (SCALE) * BASE_N_X))
    SIZE_N_Y=$((2 ** (SCALE) * BASE_N_X))
    echo "Running with ${PROC} process Grid size ${SIZE_N_X}*${SIZE_N_Y}"
    make clean
    make MAX_N_X=-DMAX_N_X=${SIZE_N_X} MAX_N_Y=-DMAX_N_Y=${SIZE_N_Y}
    mpirun -np ${PROC} ./main > benchmark.results/weak_main_${PROC}_${SIZE_N_X}*${SIZE_N_Y}
    echo ""
done    

# Strong Scaling Test
echo "Strong Scaling"
for SCALE in "${SCALE_[@]}"; do
    PROC=$((2 ** (2*SCALE)))
    echo "Running with ${PROC} process Grid size ${SIZE_N_X}*${SIZE_N_Y}"
    make clean
    make MAX_N_X=-DMAX_N_X=${SIZE_N_X} MAX_N_Y=-DMAX_N_Y=${SIZE_N_Y}
    mpirun -np ${PROC} ./main > benchmark.results/strong_main_${PROC}_${SIZE_N_X}*${SIZE_N_Y}
    echo ""
done


# Weak Scaling Test
echo "Weak Scaling with OMP"
for SCALE in "${SCALE_[@]}"; do
    PROC=$((2 ** (2*SCALE)))
    SIZE_N_X=$((2 ** (SCALE) * BASE_N_X))
    SIZE_N_Y=$((2 ** (SCALE) * BASE_N_X))
    echo "Running with ${PROC} process Grid size ${SIZE_N_X}*${SIZE_N_Y}"
    make clean
    make MAX_N_X=-DMAX_N_X=${SIZE_N_X} MAX_N_Y=-DMAX_N_Y=${SIZE_N_Y} USE_OMP=1
    mpirun -np ${PROC} ./main > benchmark.results/weak_main_omp_${PROC}_${SIZE_N_X}*${SIZE_N_Y}
    echo ""
done    

# Strong Scaling Test
echo "Strong Scaling with OMP"
for SCALE in "${SCALE_[@]}"; do
    PROC=$((2 ** (2*SCALE)))
    echo "Running with ${PROC} process Grid size ${SIZE_N_X}*${SIZE_N_Y}"
    make clean
    make MAX_N_X=-DMAX_N_X=${SIZE_N_X} MAX_N_Y=-DMAX_N_Y=${SIZE_N_Y} USE_OMP=1
    mpirun -np ${PROC} ./main > benchmark.results/strong_main_omp_${PROC}_${SIZE_N_X}*${SIZE_N_Y}
    echo ""
done
