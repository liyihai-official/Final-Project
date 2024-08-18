

#include <mpi.h>
#include <omp.h>

#include <cmath>
#include <iostream>

#include <numbers>
#include <types.hpp>

#include <pde/detials/Heat_2D.hpp>
#include <pde/detials/InitializationsC.hpp>
#include <pde/detials/BoundaryConditions/BoundaryConditions_2D.hpp>

#if !defined(STRATEGY)
#define STRATEGY final_project::PURE_MPI
#endif

#if !defined(NX) || !defined(NY)
#define NX 10000+2
#define NY 10000+2
#endif

using Integer = final_project::Integer;
using Char    = final_project::Char;
using Double  = final_project::Double;
using Float   = final_project::Float;
using size_type = final_project::Dworld;


using maintype  = Float;


using BCFunction = std::function<maintype(maintype, maintype, maintype)>;
using ICFunction = std::function<maintype(maintype, maintype)>;

#include <unistd.h>

Integer 
  main( Integer argc, Char ** argv)
{
  // constexpr final_project::Strategy strategy {final_project::PURE_MPI};

  constexpr Integer root_proc {0};
  constexpr maintype tol {1E1};
  constexpr size_type nsteps {100'000'000};
  constexpr size_type numDim {2}, nx {NX}, ny {NY};

  auto mpi_world {final_project::mpi::environment(argc, argv)};

  int opt;
  while ((opt = getopt(argc, argv, "hvf:")) != -1) 
  {
    switch (opt) {
        case 'h':
            // print_help();
                std::cout << "Program version 1.0\n";
            exit(0);
        case 'v':
            // print_version();
                std::cout << "Program version 2.0\n";
            exit(0);
        case 'f':
            // print_version();
                std::cout << "Program version f.0\n " << optarg << "\n";
            // exit(0);
            break;
        default:
            std::cerr << "Invalid option: -" << static_cast<char>(opt) << "\n";
                std::cout << "Program version HELP\n";
            exit(EXIT_FAILURE);
    }
  }



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


  switch (STRATEGY)
  {
    case final_project::PURE_MPI:
      iter = obj.solve_pure_mpi(tol, nsteps, root_proc);
      obj.reset();
      break;
    case final_project::HYBRID_0:
      iter = obj.solve_hybrid_mpi_omp(tol, nsteps, root_proc);
      obj.reset();
      break;
    case final_project::HYBRID_1:
      iter = obj.solve_hybrid2_mpi_omp(tol, nsteps, root_proc);
      obj.reset();
      break;
    default:
      std::cerr << "Undefined Parallel Strategy" << std::endl;
      break;
  }

  // obj.SaveToBinary("test.bin");
  return 0;
}