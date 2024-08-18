///
/// @file main1_3d.cpp
/// 
/// @brief
///
///
///
/// @author LI Yihai
///
///
///

/// Message Passing Interface & OpenMP 
#include <mpi.h>
#include <omp.h>

/// Standard Library Files
#include <cmath>
#include <numbers>
#include <iostream>
#include <functional>

/// Final Project header files
#include <types.hpp>
#include <helper.hpp>
/// PDE 
#include <pde/detials/Heat_3D.hpp>
#include <pde/detials/InitializationsC.hpp>
#include <pde/detials/BoundaryConditions/BoundaryConditions_3D.hpp>

/// Problem Size + Boundaries
#if !defined(NX) || !defined (NY) || !defined(NZ)
#define NX 2000+2
#define NY 2000+2
#define NZ 2000+2
#endif

/// Using datatypes
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
  Integer opt {0}, iter {0};
  final_project::Strategy strategy {final_project::Strategy::UNKNOWN};
  final_project::String filename {""};        // default empty filename (not saving results).

  constexpr Integer root_proc {0};
  constexpr maintype tol {1E1};
  constexpr size_type nsteps {100'000'000};
  constexpr size_type numDim {3}, nx {NX}, ny {NY}, nz {NZ};

  auto mpi_world {final_project::mpi::environment(argc, argv)};

  #pragma omp parallel  // Predefined Arguments
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
  std::cout << "OpenMP Threads: " << omp_get_num_threads() << "\n" << std::endl;
}
    }
  }

  while ((opt = getopt(argc, argv, "HhVvF:f:S:s:")) != -1)  // Command Line Arguments
  {
switch (opt) 
{
  case 'S': case 's':
    strategy = final_project::getStrategyfromString(optarg);
    break;
  case 'F': case 'f':
    filename = optarg;
    std::cout 
      << "This program will store results to the file: " 
      << filename << std::endl;
    break;
  case 'H': case 'h':
    final_project::helper_message(mpi_world);
    exit(EXIT_FAILURE);
  case 'V': case 'v':
    final_project::version_message(mpi_world);
    exit(EXIT_FAILURE);
  default:
    std::cerr
      << "Invalid option: -" 
      << static_cast<Char>(opt) 
      << "\n";
    exit(EXIT_FAILURE);
}
  }

  /// Heat Equation Object
  final_project::pde::Heat_3D<maintype> obj (mpi_world, nx, ny, nz);

  /// Initial Condition
  ICFunction InitCond { [](maintype x, maintype y, maintype z) { return 0; }};
  final_project::pde::InitialConditions::Init_3D<maintype> IC (InitCond);
  obj.SetHeatInitC(IC);

  /// Boundary Conditions
  final_project::pde::BoundaryConditions_3D<maintype> BC (true, true, true, true, true, true);
  BCFunction Dim000 {[](maintype x, maintype y, maintype z, maintype t){ return y + z - 2 * y * z; }};
  BCFunction Dim001 {[](maintype x, maintype y, maintype z, maintype t){ return 1 - y - z + 2 * y * z; }};

  BCFunction Dim010 {[](maintype x, maintype y, maintype z, maintype t){ return x + z - 2 * x * z; }};
  BCFunction Dim011 {[](maintype x, maintype y, maintype z, maintype t){ return 1 - x - z + 2 * x * z; }};

  BCFunction Dim100 {[](maintype x, maintype y, maintype z, maintype t){ return x + y - 2 * x * y; }};
  BCFunction Dim101 {[](maintype x, maintype y, maintype z, maintype t){ return 1 - x - y + 2 * x * y; }};
  obj.SetHeatBC(BC, Dim000, Dim001, Dim010, Dim011, Dim100, Dim101);

  /// Solving with Specified Strategy
  switch (strategy)
  {
    case final_project::Strategy::PURE_MPI:
      iter = obj.solve_pure_mpi(tol, nsteps, root_proc);
      break;
    case final_project::Strategy::HYBRID_0:
      iter = obj.solve_hybrid_mpi_omp(tol, nsteps, root_proc);
      break;
    case final_project::Strategy::HYBRID_1:
      iter = obj.solve_hybrid2_mpi_omp(tol, nsteps, root_proc);
      break;
    default:
      if (mpi_world.rank() == 0)
      {
std::cerr << "Undefined Parallel Strategy, FAIL TO SOLVE. Print Usage Message. MPI_Abort next.\n" << std::endl;
final_project::helper_message(mpi_world);
      }
      FINAL_PROJECT_MPI_ASSERT_GLOBAL(strategy == final_project::Strategy::UNKNOWN);
      break;
  }

  /// Save if needs
  if (!filename.empty()) obj.SaveToBinary(filename);
  obj.reset();

  return 0;
}