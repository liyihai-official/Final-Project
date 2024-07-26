

#include <iostream>
#include <mpi.h>

#include "types.hpp"
#include "mpi/types.hpp"


int 
main ( int argc, char ** argv )
{
  MPI_Init(&argc, &argv);
  std::cout 
    << "Running Test Program"
    << std::endl;

  final_project::Word a;

  MPI_Datatype A {final_project::mpi::get_mpi_type<final_project::Word>()};
  int size;
  MPI_Type_size(A, &size);
  std::cout << size <<std::endl;
  


  MPI_Finalize();
  return 0;
}