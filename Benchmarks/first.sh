#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2  # 每个节点启动2个MPI进程
#SBATCH --cpus-per-task=32    # 每个MPI进程分配32个CPU核心
#SBATCH -t 24:00:00
#SBATCH -o output/first.o.%j
#SBATCH -e error/first.e.%j
#SBATCH --mail-user=liy35@tcd.ie
#SBATCH --mail-type=ALL

module load openmpi/4.1.6-gcc-13.1.0-kzjsbji

export OMP_NUM_THREADS=32

mpirun --bind-to core --map-by ppr:2:node:pe=32 ../Hybrid.Heat/ver1.0/build/main1