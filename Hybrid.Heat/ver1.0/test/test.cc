

#include <iostream>
#include <mpi.h>

#include "types.hpp"

// #include "mpi/types.hpp"
// #include "mpi/assert.hpp"

// #include "mpi/environment.hpp"
// #include "mpi/topology.hpp"
// #include "mpi/multiarray.hpp"

// #include "assert.hpp"

// #include "multiarray/types.hpp"
// #include "multiarray/base.hpp"
#include "multiarray.hpp"


#if !defined(NX) || !defined(NY)
#define NX 10+2
#define NY 15+2
#endif


int 
main ( int argc, char ** argv )
{

  constexpr final_project::Integer root_proc {0};
  constexpr final_project::Double tol {1E-3};
  constexpr final_project::Dworld nsteps {1000000}, stepinterval {nsteps / 1000};
  constexpr final_project::Dworld numDIM {2}, nx {NX}, ny {NY};

  auto mpi_world {final_project::mpi::environment(argc, argv)};

  auto mat {final_project::mpi::array_Cart<double, 2>(mpi_world, nx, ny)};
  auto gather {final_project::multi_array::array_base<double, 2>(nx, ny)};
  

  // Brief information of setups
  if (root_proc == mpi_world.rank())
  {
    std::cout << numDIM << " Dimension Simulation Parameters: "     << std::endl;
    std::cout << "\tRows: "       << nx 
              << "\n\tColumns: "  << ny     << std::endl;
    std::cout << "\tTime steps: " << nsteps << std::endl;
    std::cout << "\tTolerance: "  << tol    << std::endl;

    std::cout << "MPI Parameters: "             << std::endl;
    std::cout << "\tNumber of MPI Processes: "  << mpi_world.size() << std::endl;
    std::cout << "\tRoot Process: "             << root_proc        << std::endl;
  }

  MPI_Barrier(mpi_world.comm());
  if (mpi_world.rank () == 0)
  {
    mat.array().__loc_array.fill(1);
  }
  std::cout << mat.array() << std::endl;

  
  final_project::mpi::Gather(gather, mat);
  if (mpi_world.rank() == 0)
  {
    std::cout << gather.data() << std::endl;
  }


  return 0;
}