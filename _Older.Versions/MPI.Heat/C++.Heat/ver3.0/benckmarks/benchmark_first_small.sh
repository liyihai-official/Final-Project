#!/bin/bash
#SBATCH -N 4
#SBATCH -n 256
#SBATCH -t 24:00:00
#SBATCH -J NO-OMP
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
if [ ! -d "benchmark.results.29.5.first" ]; then
    mkdir benchmark.results.29.5.first
fi

PROCS=(      64   2   4   8  16  32    1 )
# SIZE_N_X=1024
# SIZE_N_Y=1024

# echo ""
# echo "Strong Scaling NO OMP"
# echo ""
# for idx in 0 1 2 ; do
#   PROC=${PROCS[idx]}
#   echo "Running with ${PROC} process Grid size ${SIZE_N_X}*${SIZE_N_Y}"
#   make clean
#   make MAX_N_X=-DMAX_N_X=${SIZE_N_X} MAX_N_Y=-DMAX_N_Y=${SIZE_N_Y}
#   mpirun --map-by ppr:64:node -np ${PROC} ./main > benchmark.results.25.5.first/strong_main_${PROC}_${SIZE_N_X}*${SIZE_N_Y}
#   echo ""
# done

SIZE_N_X=256
SIZE_N_Y=256
SIZE_N_Z=256

echo ""
echo "Strong Scaling NO OMP"
echo ""
for idx in 0 1 2 3 4 5 6; do
  PROC=${PROCS[idx]}
  echo "Running with ${PROC} process Grid size ${SIZE_N_X}*${SIZE_N_Y}*${SIZE_N_Z}"
  make clean
  make MAX_N_X=-DMAX_N_X=${SIZE_N_X} MAX_N_Y=-DMAX_N_Y=${SIZE_N_Y} MAX_N_Z=-DMAX_N_Z=${SIZE_N_Z}
  mpirun --map-by ppr:64:node -np ${PROC} ./main_3d > benchmark.results.29.5.first/strong_main_3d_${PROC}_${SIZE_N_X}*${SIZE_N_Y}*${SIZE_N_Z}
  echo ""
done


# PROCS=(        1   4   9  16  25  36   49   64  )
# SIZE_N_XS=(  128 256 384 512 630 756  882 1024  ) # 128 128 256 256 512 512  1024
# SIZE_N_YS=(  128 256 384 512 630 756  882 1024  ) # 128 256 256 512 512 1024 1024 

# echo "Weak Scaling NO OMP"
# echo ""
# for idx in 0 1 2 3 4 5 6 7 8; do
#   PROC=${PROCS[idx]}
#   SIZE_N_X=$((SIZE_N_XS[idx]))
#   SIZE_N_Y=$((SIZE_N_YS[idx]))
#   echo "Running with ${PROC} process Grid size ${SIZE_N_X}*${SIZE_N_Y}"
#   make clean
#   make MAX_N_X=-DMAX_N_X=${SIZE_N_X} MAX_N_Y=-DMAX_N_Y=${SIZE_N_Y}
#   mpirun --map-by ppr:64:node -np ${PROC} ./main > benchmark.results.25.5.first/weak_main_${PROC}_${SIZE_N_X}*${SIZE_N_Y}
#   echo ""
# done
