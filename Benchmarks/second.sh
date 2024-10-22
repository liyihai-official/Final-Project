#!/bin/bash
#SBATCH --nodes=4
#SBATCH -t 24:00:00
#SBATCH -o output/second.o.%j
#SBATCH -e error/second.e.%j
#SBATCH --mail-user=liy35@tcd.ie
#SBATCH --mail-type=ALL

cat second.sh

module load openmpi/4.1.6-gcc-13.1.0-kzjsbji

export OMP_NUM_THREADS=32

mpirun --bind-to core --map-by ppr:4:node:pe=32 ../Hybrid.Heat/ver1.0/build/main1

git add output/*
git add error/*
git commit -m "Second.2D.Hybrid.Test"
git push