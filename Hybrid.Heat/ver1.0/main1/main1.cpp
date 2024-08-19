///
/// @file main1.cpp
///
/// @brief
///
///
///
/// @link https://github.com/liyihai-official/Final-Project
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
#include <pde/detials/Heat_2D.hpp>
#include <pde/detials/InitializationsC.hpp>
#include <pde/detials/BoundaryConditions/BoundaryConditions_2D.hpp>

/// Problem Size + Boundaries
#if !defined(NX) || !defined(NY)
#define NX 100+2
#define NY 100+2
#endif

/// Using datatypes
using Integer = final_project::Integer;
using Char    = final_project::Char;
using Double  = final_project::Double;
using Float   = final_project::Float;
using size_type = final_project::Dworld;

using maintype  = Float;

using BCFunction = std::function<maintype(maintype, maintype, maintype)>;
using ICFunction = std::function<maintype(maintype, maintype)>;


Integer 
  main( Integer argc, Char ** argv)
{
  Integer opt {0}, iter {0};
  final_project::Strategy strategy {final_project::Strategy::UNKNOWN};
  final_project::String filename {""};        // default empty filename (not saving results).

  constexpr Integer root_proc {0};
  constexpr maintype tol {1E-1};
  constexpr size_type nsteps {100'000'000};
  constexpr size_type numDim {2}, nx {NX}, ny {NY};

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
    if (mpi_world.rank() == 0)
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
  final_project::pde::Heat_2D<maintype> obj (mpi_world, nx, ny);

  /// Initial Condition
  ICFunction InitCond {[](maintype x, maintype y) { return 0; }};
  final_project::pde::InitialConditions::Init_2D<maintype> IC (InitCond);
  obj.SetHeatInitC(IC);
  
  /// Boundary Conditions
  final_project::pde::BoundaryConditions_2D<maintype> BC (true, true, true, true);

  BCFunction Dim00 {[](maintype x, maintype y, maintype t){ return y;}};
  BCFunction Dim01 {[](maintype x, maintype y, maintype t){ return 1;}};

  BCFunction Dim10 {[](maintype x, maintype y, maintype t){ return x;}};
  BCFunction Dim11 {[](maintype x, maintype y, maintype t){ return 1;}};
  obj.SetHeatBC(BC, Dim00, Dim01, Dim10, Dim11);

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