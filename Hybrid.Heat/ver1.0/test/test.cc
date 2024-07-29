

#include <iostream>
#include <mpi.h>

#include "types.hpp"

#include "mpi/types.hpp"
#include "mpi/assert.hpp"

#include "mpi/environment.hpp"
#include "mpi/topology.hpp"

#include "assert.hpp"

#include "multiarray/types.hpp"
#include "multiarray/base.hpp"
#include "multiarray.hpp"


int 
main ( int argc, char ** argv )
{
  auto env = final_project::mpi::environment(argc, argv);
  // MPI_Init(&argc, &argv);
  // auto B = env.size();
  // std::cout << B;
  

  // // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::cout 
    << "Running Test Program"
    << std::endl;

  final_project::Word a;

  MPI_Datatype A {final_project::mpi::get_mpi_type<final_project::Integer>()};
  MPI_Datatype B {final_project::mpi::get_mpi_type<final_project::Dworld>()};
  int sizeA, sizeB;
  MPI_Type_size(A, &sizeA);
  MPI_Type_size(B, &sizeB);

  std::cout << sizeA << "\t" << sizeB << std::endl;
  

  auto SS {final_project::multi_array::__detail::__multi_array_shape<2>(4,7)};
  // std::cout << " [" << SS[0] 
  //           << ", " << SS[1] 
  //           << ", " << SS[2] 
  //           << "] " << std::endl;

  auto Mat {final_project::multi_array::__detail::__array<double, 2>(SS)};
  Mat.fill(1);
  // std::cout << Mat << std::endl;

  final_project::mpi::topology::Cartesian<final_project::Double, 2> togo(SS, env);



  // MPI_Finalize();
  return 0;
}