#!/bin/bash
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 48:00:00
#SBATCH -o output/forth.o.%j
#SBATCH -e error/forth.e.%j
#SBATCH --mail-user=liy35@tcd.ie
#SBATCH --mail-type=ALL

module load openmpi/4.1.6-gcc-13.1.0-kzjsbji

export OMP_NUM_THREADS=1

echo "Strong Scalling Tests, NO OpenMP, 3D, Pure MPI"
echo ""

for PROC in 64 32 16 8 4 2 1; do
  mpirun -np ${PROC} ../Hybrid.Heat/ver1.0/build/main1_3d
done


git add output/*
git add error/*
git commit -m "Forth.3D.Pure.MPI.Test"
git push