

#include <mpi.h>
#include <omp.h>

#include <cmath>
#include <iostream>

#include <numbers>
#include <types.hpp>

#include <pde/detials/Heat_2D.hpp>
#include <pde/detials/InitializationsC.hpp>
#include <pde/detials/BoundaryConditions/BoundaryConditions_2D.hpp>


#if !defined(NX) || !defined(NY)
#define NX 98+2
#define NY 98+2
#endif


using Integer = final_project::Integer;
using Char    = final_project::Char;
using Double  = final_project::Double;
using Float   = final_project::Float;
using size_type = final_project::Dworld;

using BCFunction = std::function<Float(Float, Float, Float)>;
using ICFunction = std::function<Float(Float, Float)>;

Integer 
  main( Integer argc, Char ** argv)
{

  constexpr Integer root_proc {0};
  constexpr Float tol {1E-4};
  constexpr size_type nsteps {10000};
  constexpr size_type numDim {2}, nx {NX}, ny {NY};

  auto mpi_world {final_project::mpi::environment(argc, argv)};

  final_project::pde::Heat_2D<Float> obj (mpi_world, nx, ny);

  ICFunction InitCond {[](Float x, Float y) { return 0; }};
  final_project::pde::InitialConditions::Init_2D<Float> IC (InitCond);
  
  final_project::pde::BoundaryConditions_2D<Float> BC (true, true, true, true);

  BCFunction Dim00 {[](Float x, Float y, Float t){ return 5 * std::abs(std::sin(y * 2 * std::numbers::pi));}};
  BCFunction Dim01 {[](Float x, Float y, Float t){ return 0;}};

  BCFunction Dim10 {[](Float x, Float y, Float t){ return 20 * std::sin(x * 2 * std::numbers::pi);}};
  BCFunction Dim11 {[](Float x, Float y, Float t){ return -20  * std::sin(x * 2 * std::numbers::pi);}};

  obj.SetHeatBC(BC, Dim00, Dim01, Dim10, Dim11);
  obj.SetHeatInitC(IC);

  // auto iter = obj.solve_pure_mpi(tol, nsteps, root_proc);
  auto iter = obj.solve_hybrid_mpi_omp(tol, nsteps, root_proc);
  // auto iter = obj.solve_hybrid2_mpi_omp(tol, nsteps, root_proc);

  obj.SaveToBinary("test.bin");


  return 0;
}