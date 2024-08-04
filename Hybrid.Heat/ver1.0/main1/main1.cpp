

#include <mpi.h>
#include <iostream>

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

Integer 
  main( Integer argc, Char ** argv)
{

  constexpr Integer root_proc {0};
  constexpr Double tol {1E-3};
  constexpr size_type nsteps {100'000};
  constexpr size_type numDim {2}, nx {NX}, ny {NY};

  auto mpi_world {final_project::mpi::environment(argc, argv)};

  final_project::pde::Heat_2D<Double> obj (mpi_world, nx, ny);

  auto InitCond = [](Double x, Double y) { return x + y; };
  final_project::pde::InitialConditions::Init_2D<Double> IC (InitCond);
  
  final_project::pde::BoundaryConditions_2D<Double> BC (false, true, false, true);

  obj.SetHeatBC(BC, -1, 2, 1, 4);
  obj.SetHeatInitC(IC);

  auto iter = obj.solve_pure_mpi(tol, nsteps, root_proc);

  obj.SaveToBinary("test.bin");

  return 0;
}