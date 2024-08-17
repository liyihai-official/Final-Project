

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
#define NX 10+2
#define NY 8+2
#endif


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

  constexpr Integer root_proc {0};
  constexpr maintype tol {1E-1};
  constexpr size_type nsteps {100'000'000};
  constexpr size_type numDim {2}, nx {NX}, ny {NY};

  auto mpi_world {final_project::mpi::environment(argc, argv)};

  Integer iter {0};
  final_project::pde::Heat_2D<maintype> obj (mpi_world, nx, ny);

  ICFunction InitCond {[](maintype x, maintype y) { return 0; }};
  final_project::pde::InitialConditions::Init_2D<maintype> IC (InitCond);
  
  final_project::pde::BoundaryConditions_2D<maintype> BC (true, true, true, true);

  BCFunction Dim00 {[](maintype x, maintype y, maintype t){ return y;}};
  BCFunction Dim01 {[](maintype x, maintype y, maintype t){ return 1;}};

  BCFunction Dim10 {[](maintype x, maintype y, maintype t){ return x;}};
  BCFunction Dim11 {[](maintype x, maintype y, maintype t){ return 1;}};

  obj.SetHeatBC(BC, Dim00, Dim01, Dim10, Dim11);
  obj.SetHeatInitC(IC);

  iter = obj.solve_pure_mpi(tol, nsteps, root_proc);

  obj.reset();
  iter = obj.solve_hybrid_mpi_omp(tol, nsteps, root_proc);

  obj.reset();
  iter = obj.solve_hybrid2_mpi_omp(tol, nsteps, root_proc);

  obj.SaveToBinary("test.bin");
  return 0;
}