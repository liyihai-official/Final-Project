#!/bin/bash
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 48:00:00
#SBATCH -o output/forth.o.%j
#SBATCH -e error/forth.e.%j
#SBATCH --mail-user=liy35@tcd.ie
#SBATCH --mail-type=ALL

# Load Essential Module / PATH
module load openmpi/4.1.6-gcc-13.1.0-kzjsbji
export LIBTORCH_PATH=/home/users/mschpc/2023/liy35/libtorch:$LIBTORCH_PATH

echo "Strong Scalling Tests, NO OpenMP, 3D, Pure MPI"
NX=8002
NY=8002
NZ=2
export OMP_NUM_THREADS=1
export NX=${NX} NY=${NY} NZ=${NZ}
echo ""

# cmake -DCMAKE_PREFIX_PATH=/home/users/mschpc/2023/liy35/libtorch -DCMAKE_BUILD_TYPE=DEBUG -DNX=8002 -DNY=8002 -DNZ=4 ..
# for PROC in 64 32 16 8 4 2 1; do
#   echo "\n Strong Test : ${PROC} \n"
#   mpirun -np ${PROC} ../Hybrid.Heat/ver1.0/build/main1_3d
# done


# git add output/*
# git add error/*
# git commit -m "Forth.3D.Pure.MPI.Test"
# git push