

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
#define NX 20000+2
#define NY 20000+2
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
  constexpr Float tol {1E3};
  constexpr size_type nsteps {100'000'000};
  constexpr size_type numDim {2}, nx {NX}, ny {NY};

  auto mpi_world {final_project::mpi::environment(argc, argv)};

  final_project::pde::Heat_2D<Float> obj (mpi_world, nx, ny);

  ICFunction InitCond {[](Float x, Float y) { return 0; }};
  final_project::pde::InitialConditions::Init_2D<Float> IC (InitCond);
  
  final_project::pde::BoundaryConditions_2D<Float> BC (true, true, true, true);

  BCFunction Dim00 {[](Float x, Float y, Float t){ return y;}};
  BCFunction Dim01 {[](Float x, Float y, Float t){ return 1-y;}};

  BCFunction Dim10 {[](Float x, Float y, Float t){ return x;}};
  BCFunction Dim11 {[](Float x, Float y, Float t){ return 1-x;}};

  obj.SetHeatBC(BC, Dim00, Dim01, Dim10, Dim11);
  obj.SetHeatInitC(IC);

  auto iter = obj.solve_pure_mpi(tol, nsteps, root_proc);
  // auto iter = obj.solve_hybrid_mpi_omp(tol, nsteps, root_proc);
  // auto iter = obj.solve_hybrid2_mpi_omp(tol, nsteps, root_proc);

  obj.SaveToBinary("test.bin");

  std::cout << iter << std::endl;
  return 0;
}