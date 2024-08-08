

#include <mpi.h>
#include <omp.h>

#include <cmath>
#include <iostream>

#include <numbers>
#include <types.hpp>

#include <pde/detials/Heat_2D.hpp>
#include <pde/detials/InitializationsC.hpp>
#include <pde/detials/BoundaryConditions/BoundaryConditions.hpp>


#if !defined(NX) || !defined(NY)
#define NX 1000+2
#define NY 1000+2
#endif


using Integer = final_project::Integer;
using Char    = final_project::Char;
using Double  = final_project::Double;
using size_type = final_project::Dworld;

using BCFunction = std::function<Double(Double, Double, Double)>;
using ICFunction = std::function<Double(Double, Double)>;

Integer 
  main( Integer argc, Char ** argv)
{

  constexpr Integer root_proc {0};
  constexpr Double tol {1E0};
  constexpr size_type nsteps {100'000'000};
  constexpr size_type numDim {2}, nx {NX}, ny {NY};

  auto mpi_world {final_project::mpi::environment(argc, argv)};

  final_project::pde::Heat_2D<Double> obj (mpi_world, nx, ny);

  ICFunction InitCond {[](Double x, Double y) { return 10; }};
  final_project::pde::InitialConditions::Init_2D<Double> IC (InitCond);
  
  final_project::pde::BoundaryConditions_2D<Double> BC (true, true, false, false);

  BCFunction Dim00 {[](Double x, Double y, Double t){ return 5 * std::abs(std::sin(y * 2 * std::numbers::pi));}};
  BCFunction Dim01 {[](Double x, Double y, Double t){ return 0;}};

  BCFunction Dim10 {[](Double x, Double y, Double t){ return 20 * std::sin(x * 2 * std::numbers::pi);}};
  BCFunction Dim11 {[](Double x, Double y, Double t){ return -20  * std::sin(x * 2 * std::numbers::pi);}};

  obj.SetHeatBC(BC, Dim00, Dim01, Dim10, Dim11);
  obj.SetHeatInitC(IC);


  // auto iter = obj.solve_pure_mpi(tol, nsteps, root_proc);
  // auto iter = obj.solve_hybrid_mpi_omp(tol, nsteps, root_proc);
  auto iter = obj.solve_hybrid2_mpi_omp(tol, nsteps, root_proc);


  obj.SaveToBinary("test.bin");


  return 0;
}