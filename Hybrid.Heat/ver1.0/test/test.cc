

#include <iostream>
#include <mpi.h>

#include "types.hpp"

#include "multiarray.hpp"

// #include <pde/pde.hpp>
// #include "pde/Heat.hpp"
#include <pde/detials/Heat_2D.hpp>
#include <pde/detials/InitializationsC.hpp>
#include <pde/detials/BoundaryConditions.hpp>



#if !defined(NX) || !defined(NY)
#define NX 8+2
#define NY 9+2
#define NZ 10+2
#endif


int 
main ( int argc, char ** argv )
{

  constexpr final_project::Integer root_proc {0};
  constexpr final_project::Double tol {1E-3};
  constexpr final_project::Dworld nsteps {1000000}, stepinterval {nsteps / 1000};
  constexpr final_project::Dworld numDim {3}, nx {NX}, ny {NY}, nz {NZ};

  auto mpi_world {final_project::mpi::environment(argc, argv)};

  final_project::pde::Heat_2D<double> obj(mpi_world, nx, ny);


  auto customLambda = [](double x, double y) { return x + y; };
  final_project::pde::InitialConditions::Init_2D<double> init(obj, 
    [](double x, double y) { return 0; }
  );

  final_project::pde::BoundaryConditions_2D<double> BC (obj, 1,2,3,4);


  // for (final_project::Integer i = 1; i < 100; ++i)
  // {
  //   obj.update_ping_pong();
  //   obj.exchange_ping_pong();
  //   obj.switch_in_out();
  // }
  // obj.show();

  // auto gather {final_project::multi_array::array_base<double, 2>(nx, ny)};

  // std::cout << A << std::endl;

  obj.solve_pure_mpi(1E-3);



  // auto mat {final_project::mpi::array_Cart<double, numDim>(mpi_world, nx, ny, nz)};
  // auto gather {final_project::multi_array::array_base<double, numDim>(nx, ny, nz)};
  

  // // Brief information of setups
  // if (root_proc == mpi_world.rank())
  // {
  //   std::cout << numDim << " Dimension Simulation Parameters: "     << std::endl;
  //   std::cout << "\tRows: "       << nx 
  //             << "\n\tColumns: "  << ny     << std::endl;
  //   std::cout << "\tTime steps: " << nsteps << std::endl;
  //   std::cout << "\tTolerance: "  << tol    << std::endl;

  //   std::cout << "MPI Parameters: "             << std::endl;
  //   std::cout << "\tNumber of MPI Processes: "  << mpi_world.size() << std::endl;
  //   std::cout << "\tRoot Process: "             << root_proc        << std::endl;
  // }

  // MPI_Barrier(mpi_world.comm());
  // if (mpi_world.rank () == 0)
  // {
  //   gather.data().fill(-1);
  // }
  // mat.array().__loc_array.fill(5+mpi_world.rank());

  
  // final_project::mpi::Gather(gather, mat, 0);


  // MPI_Barrier(mpi_world.comm());
  // if (mpi_world.rank() == 0)
  // {
  //   std::cout << gather.data() << std::endl;
  // }

  // std::cout << mat.array() << std::endl;

  return 0;
}