#include <iostream>
#include <memory>
#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

int main( int argc, char ** argv)
{
  MPI_Init(&argc, &argv);
  int num_threads = 1, rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);


  #pragma omp parallel
  {
    #ifdef _OPENMP
      #pragma omp master
      num_threads = omp_get_num_threads();
    #endif

    #pragma omp master
    std::cout << "MPI PROC: " 
              << rank << "/" << size
              << "\tOpenMP has : " 
              << num_threads 
              << " threads."  
              << std::endl;
  }

  MPI_Finalize();
  return 0;
}