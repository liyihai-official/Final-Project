#!/bin/bash
#SBATCH -N 1
#SBATCH -n 96
#SBATCH --mail-user=liy35@tcd.ie
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=96
#SBATCH --partition=9242
#SBATCH --output=output/single.o.%j
#SBATCH --error=error/single.e.%j

module load cmake/3.26.5
module load openmpi/4.1.5-gcc13.2.0
module load gcc/13.2.0

export OMPI_MCA_btl_openib_allow_ib=1

export PATH_TO_GCC=/data/app/gcc13.2.0/bin/gcc
export PATH_TO_GXX=/data/app/openmpi/4.1.5-gcc13.2.0/bin/mpicxx
export PATH_TO_LIBTORCH=/data/home/pokemonarrive_gmail.com/libtorch


scale_values=(32 64 128 256)

ppr_values=(1 2 4 8 16 24 32 48 64 96)

ppnuma_values=(1 2 4)
thread_values=(1 2 4 8 16 24)


# Runs
for run in {1..1}; do
  echo "Run #$run"

# Compile
  for scale in "${scale_values[@]}"; do
    cmake -B build -S ../Hybrid.Heat/ver1.0/ -DCMAKE_C_COMPILER=${PATH_TO_GCC} -DCMAKE_CXX_COMPILER=${PATH_TO_GXX} -DCMAKE_PREFIX_PATH=${PATH_TO_LIBTORCH} -DCMAKE_BUILD_TYPE=Release -DNX=${scale} -DNY=${scale} -DNZ=${scale}
    cmake --build build --target main1_3d -j16

# Pure MPI
    for ppr in "${ppr_values[@]}"; do
      echo "Strategy: pure_mpi"
      mpirun --map-by ppr:$ppr:node:pe=1 --report-bindings build/main1_3d -S PURE_MPI
      echo "====================================================================="
    done

# Hybrid 0
    for ppr in "${ppnuma_values[@]}"; do
      for thread in "${thread_values[@]}"; do
        echo "Strategy: hybrid_0"
        mpirun --map-by ppr:$ppr:numa:pe=$thread --report-bindings build/main1_3d -S HYBRID_0
        echo "====================================================================="
      done  
    done

# Hybrid 1
    for ppr in "${ppnuma_values[@]}"; do
      for thread in "${thread_values[@]}"; do
        echo "Strategy: hybrid_1"      
        mpirun --map-by ppr:$ppr:numa:pe=$thread --report-bindings build/main1_3d -S HYBRID_1
        echo "====================================================================="
      done
    done

done