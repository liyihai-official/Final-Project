#!/bin/bash
#SBATCH -N 4
#SBATCH -n 256
#SBATCH -t 24:00:00
#SBATCH -J gperftest
#SBATCH -o benchmark_first.o.%j
#SBATCH -e benchmark_first.e.%j
#SBATCH --mail-user=liy35@tcd.ie
#SBATCH --mail-type=ALL
#SBATCH --account=callan_liy35
#SBATCH --partition=compute

# File: benchmark.sh
# Create the benchmark.results directory if it doesn't exist
if [ ! -d "gperftools.results" ]; then
    mkdir gperftools.results
fi

# Load Enssential Module
module load openmpi/4.1.6-gcc-13.1.0-kzjsbji
module load gperftools/

# Load Enssential Environmental Variables
export LIBRARY_PATH=/home/support/spack/0.21.1/opt/spack/linux-rocky8-sapphirerapids/gcc-13.1.0/gperftools-2.13-542of25xw4qrr4nbzdrbxfdegsu7ryd2/lib:$LIBRARY_PATH
export CPATH=/home/support/spack/0.21.1/opt/spack/linux-rocky8-sapphirerapids/gcc-13.1.0/gperftools-2.13-542of25xw4qrr4nbzdrbxfdegsu7ryd2/include:$CPATH
export LD_LIBRARY_PATH=/home/support/spack/0.21.1/opt/spack/linux-rocky8-sapphirerapids/gcc-13.1.0/gperftools-2.13-542of25xw4qrr4nbzdrbxfdegsu7ryd2/lib:$LD_LIBRARY_PATH

# Set the Dimensions (Large Enough to get samples)
N_X=4096
N_Y=4096

PROCS=128

# Tests
make clean
make MAX_N_X=-DMAX_N_X=${N_X} MAX_N_Y=-DMAX_N_Y=${N_Y}

mpiexec -np ${PROCS} -x CPUPROFILE='gperftools.results/main.prof' ./main

for i in $(seq 0 $((PROCS - 1)))
do
  pprof --svg ./main gperftools.results/main.prof.rank-${i} > gperftools.results/main.rank-${i}.svg
  pprof --pdf ./main gperftools.results/main.prof.rank-${i} > gperftools.results/main.rank-${i}.pdf
  rm gperftools.results/main.prof.rank-${i}
done

make clean