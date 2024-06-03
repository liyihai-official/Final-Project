#!/bin/bash
#SBATCH -N 2
#SBATCH -n 128
#SBATCH -t 24:00:00
#SBATCH -J NO-OMP
#SBATCH -o benchmark_second.o.%j
#SBATCH -e benchmark_second.e.%j
#SBATCH --mail-user=liy35@tcd.ie
#SBATCH --mail-type=ALL
#SBATCH --account=callan_liy35
#SBATCH --partition=compute

# Load Enssential Module
module load openmpi/4.1.6-gcc-13.1.0-kzjsbji

# File: benchmark.sh
# Create the benchmark.results directory if it doesn't exist
if [ ! -d "benchmark.results.23.5.second" ]; then
    mkdir benchmark.results.23.5.second
fi


PROCS=(      1   2   4   8  16  32    64 128)
SIZE_N_X=1024
SIZE_N_Y=1024

echo ""
echo "Strong Scaling NO OMP"
echo ""
for idx in 0 1 2 3 4 5 6; do
  PROC=${PROCS[idx]}
  echo "Running with ${PROC} process Grid size ${SIZE_N_X}*${SIZE_N_Y}"
  make clean
  make MAX_N_X=-DMAX_N_X=${SIZE_N_X} MAX_N_Y=-DMAX_N_Y=${SIZE_N_Y}
  mpirun --map-by ppr:64:node -np ${PROC} ./main > benchmark.results.23.5.second/strong_main_${PROC}_${SIZE_N_X}*${SIZE_N_Y}
  echo ""
done


PROCS=(      1   2   4   8  16  32    64  128)
SIZE_N_XS=(128 128 256 256 512 512  1024 1024)
SIZE_N_YS=(128 256 256 512 512 1024 1024 2048)

echo "Weak Scaling NO OMP"
echo ""
for idx in 0 1 2 3 4 5 6; do
  PROC=${PROCS[idx]}
  SIZE_N_X=$((SIZE_N_XS[idx]))
  SIZE_N_Y=$((SIZE_N_YS[idx]))
  echo "Running with ${PROC} process Grid size ${SIZE_N_X}*${SIZE_N_Y}"
  make clean
  make MAX_N_X=-DMAX_N_X=${SIZE_N_X} MAX_N_Y=-DMAX_N_Y=${SIZE_N_Y}
  mpirun --map-by ppr:64:node -np ${PROC} ./main > benchmark.results.23.5.second/weak_main_${PROC}_${SIZE_N_X}*${SIZE_N_Y}
  echo ""
done