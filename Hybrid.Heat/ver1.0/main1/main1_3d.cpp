#include <mpi.h>
#include <omp.h>

#include <cmath>
#include <iostream>

#include <numbers>
#include <types.hpp>

#include <pde/detials/Heat_3D.hpp>
#include <pde/detials/InitializationsC.hpp>
#include <pde/detials/BoundaryConditions/BoundaryConditions_3D.hpp>

#include <functional>
#include <mpi/environment.hpp>

#if !defined(NX) || !defined (NY) || !defined(NZ)
#define NX 2000+2
#define NY 2000+2
#define NZ 2000+2
#endif

using Integer = final_project::Integer;
using Char    = final_project::Char;
using Double  = final_project::Double;
using Float   = final_project::Float;
using size_type = final_project::Dworld;



using maintype  = Double;

using BCFunction = std::function<maintype(maintype, maintype, maintype, maintype)>;
using ICFunction = std::function<maintype(maintype, maintype, maintype)>;


Integer 
  main ( Integer argc, Char ** argv )
{

  constexpr Integer root_proc {0};
  constexpr maintype tol {1E1};
  constexpr size_type nsteps {100'000'000};
  constexpr size_type numDim {3}, nx {NX}, ny {NY}, nz {NZ};

  auto mpi_world {final_project::mpi::environment(argc, argv)};

  #pragma omp parallel
  {
    #pragma omp master
    {
      if (mpi_world.rank() == 0)
      {
        std::cout << "Problem size: " 
                  << "\n\tRows: "     << nx-2 
                  << "\n\tColumns: "  << ny-2       << std::endl;
        std::cout << "MPI Parameters: "             << std::endl;
        std::cout << "\tNumber of MPI Processes: "  << mpi_world.size()  << std::endl;
        std::cout << "OpenMP Threads: " << omp_get_num_threads()         << std::endl;
      }
    }
  }

  ///
  /// TODO:
  ///   Complete Heat_3D<T> class
  ///
  final_project::pde::Heat_3D<maintype> obj (mpi_world, nx, ny, nz);
  ICFunction InitCond { [](maintype x, maintype y, maintype z) { return 0; }};

  Integer iter {0};
  /// FINISH 
  /// TODO: 
  ///   Complete Init_3D<T> class
  ///
  final_project::pde::InitialConditions::Init_3D<maintype> IC (InitCond);

  ///
  /// TODO:
  ///
  final_project::pde::BoundaryConditions_3D<maintype> BC (true, true, true, true, true, true);
  BCFunction Dim000 {[](maintype x, maintype y, maintype z, maintype t){ return y + z - 2 * y * z; }};
  BCFunction Dim001 {[](maintype x, maintype y, maintype z, maintype t){ return 1 - y - z + 2 * y * z; }};

  BCFunction Dim010 {[](maintype x, maintype y, maintype z, maintype t){ return x + z - 2 * x * z; }};
  BCFunction Dim011 {[](maintype x, maintype y, maintype z, maintype t){ return 1 - x - z + 2 * x * z; }};

  BCFunction Dim100 {[](maintype x, maintype y, maintype z, maintype t){ return x + y - 2 * x * y; }};
  BCFunction Dim101 {[](maintype x, maintype y, maintype z, maintype t){ return 1 - x - y + 2 * x * y; }};

  obj.SetHeatInitC(IC);
  obj.SetHeatBC(BC, Dim000, Dim001, Dim010, Dim011, Dim100, Dim101);


  iter = obj.solve_pure_mpi(tol, nsteps, root_proc);

  // obj.reset();
  // iter = obj.solve_hybrid_mpi_omp(tol, nsteps, root_proc);

  // obj.reset();
  // iter = obj.solve_hybrid2_mpi_omp(tol, nsteps, root_proc);

  return 0;
}