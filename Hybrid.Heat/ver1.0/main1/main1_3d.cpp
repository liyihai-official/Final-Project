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
#define NX 98+2
#define NY 98+2
#define NZ 98+2
#endif

using Integer = final_project::Integer;
using Char    = final_project::Char;
using Double  = final_project::Double;
using Float   = final_project::Float;
using size_type = final_project::Dworld;

using BCFunction = std::function<Float(Float, Float, Float, Float)>;
using ICFunction = std::function<Float(Float, Float, Float)>;



Integer 
  main ( Integer argc, Char ** argv )
{

  constexpr Integer root_proc {0};
  constexpr Float tol {1E-4};
  constexpr size_type nsteps {2};
  constexpr size_type numDim {3}, nx {NX}, ny {NY}, nz {NZ};

  auto mpi_world {final_project::mpi::environment(argc, argv)};

  ///
  /// TODO:
  ///   Complete Heat_3D<T> class
  ///
  final_project::pde::Heat_3D<Float> obj (mpi_world, nx, ny, nz);

  ICFunction InitCond { [](Float x, Float y, Float z) { return 0; }};

  ///
  /// TODO:
  ///   Complete Init_3D<T> class
  ///
  final_project::pde::InitialConditions::Init_3D<Float> IC (InitCond);

  ///
  /// TODO:
  ///
  final_project::pde::BoundaryConditions_3D<Float> BC (true, true, true, true, true, true);
  BCFunction Dim000 {[](Float x, Float y, Float z, Float t){ return 0;}};
  BCFunction Dim001 {[](Float x, Float y, Float z, Float t){ return 0;}};

  BCFunction Dim010 {[](Float x, Float y, Float z, Float t){ return 0;}};
  BCFunction Dim011 {[](Float x, Float y, Float z, Float t){ return 0;}};

  BCFunction Dim100 {[](Float x, Float y, Float z, Float t){ return 0;}};
  BCFunction Dim101 {[](Float x, Float y, Float z, Float t){ return 0;}};

  // obj.SetHeatBC(BC, Dim000, Dim001, Dim010, Dim011, Dim100, Dim101);
  // obj.SetHeatInitC(IC);

  auto iter = obj.solve_pure_mpi(tol, nsteps, root_proc);
  // auto iter = obj.solve_hybrid_mpi_omp(tol, nsteps, root_proc);
  // auto iter = obj.solve_hybrid2_mpi_omp(tol, nsteps, root_proc);

  obj.SaveToBinary("test.bin");

  return 0;
}