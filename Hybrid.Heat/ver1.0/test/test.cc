

#include <iostream>
#include <mpi.h>

#include "types.hpp"

#include "mpi/types.hpp"
#include "mpi/assert.hpp"

#include "assert.hpp"

int 
main ( int argc, char ** argv )
{
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::cout 
    << "Running Test Program"
    << std::endl;

  final_project::Word a;

  MPI_Datatype A {final_project::mpi::get_mpi_type<final_project::Word>()};
  int size;
  MPI_Type_size(A, &size);
  

  size = 10;
  if (rank == 0) 
  {
    size = 1;
  }
  std::cout << size << "\t" << std::endl;

  // FINAL_PROJECT_MPI_ASSERT((size == 10));
  // FINAL_PROJECT_MPI_WARN((size == 10));
  // FINAL_PROJECT_MPI_ASSERT_GLOBAL((size == 10));

  
  MPI_Finalize();
  return 0;
}