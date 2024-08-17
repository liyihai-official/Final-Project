#!/bin/bash
#SBATCH --nodes=2
#SBATCH -t 24:00:00
#SBATCH -o benchmark_first.o.%j
#SBATCH -e benchmark_first.e.%j
#SBATCH --mail-user= liy35@tcd.ie (mailto:liy35@tcd.ie)
#SBATCH --mail-type=ALL

module load openmpi/4.1.6-gcc-13.1.0-kzjsbji

export OMP_NUM_THREADS=32

