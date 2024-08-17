#!/bin/bash
#SBATCH --nodes=2
#SBATCH -t 24:00:00
#SBATCH -o output/third.o.%j
#SBATCH -e error/third.e.%j
#SBATCH --mail-user=liy35@tcd.ie
#SBATCH --mail-type=ALL

module load openmpi/4.1.6-gcc-13.1.0-kzjsbji

export OMP_NUM_THREADS=32

mpirun --bind-to core --map-by ppr:2:node:pe=32 ../Hybrid.Heat/ver1.0/build/main1

