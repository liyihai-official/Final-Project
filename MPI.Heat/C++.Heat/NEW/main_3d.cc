/**
 * @file main_3d.cc
 * 
 * @brief Main file for the parallel Heat Equation solver using MPI.
 * 
 * This file sets up the MPI environment, initializes the arrays, performs
 * the heat equation computation, exchanges data between processes, gathers
 * the final results, and outputs the results.
 * 
 * @author Li Yihai
 * @version 3.0
 * @date May 28, 2024
 * 
 * @section DESCRIPTION
 * The main function initializes the MPI environment and sets up a Cartesian
 * topology for process communication. It then initializes the arrays, sets
 * initial conditions, and performs the heat equation computation using a
 * distributed approach. The results are gathered and printed by the root
 * process.
 */
#include <mpi.h>

#include "final_project.cpp"

#if !defined(MAX_N_X) || !defined(MAX_N_Y) || !defined(MAX_N_Z)
#define MAX_N_X 50+2
#define MAX_N_Y 50+2
#define MAX_N_Z 50+2
#endif

#if !defined (MAX_it)
#define MAX_it 2'000'000
#endif

#define tol 1E-13

int main ( int argc, char ** argv)
{  
  auto world = mpi::env(argc, argv);

  int i;
  double loc_diff, glob_diff {10}, t1, t2;
  constexpr int reorder {1}, dimension {3}, root {0};

  int dims[dimension], coords[dimension], periods[dimension];
  for (short int i = 0; i < dimension; ++i) {dims[i] = 0; periods[i] = 0;}

  /* MPI Cartesian Inits */
  MPI_Comm comm_cart;
  MPI_Dims_create(world.size(), dimension, dims);
  MPI_Cart_create(MPI_COMM_WORLD, dimension, dims, periods, reorder, &comm_cart);

  final_project::array3d<double> gather(MAX_N_X, MAX_N_Y, MAX_N_Z);
  final_project::array3d_distribute<double> A, B;

  A.distribute(MAX_N_X, MAX_N_Y, MAX_N_Z, dims, comm_cart);
  B.distribute(MAX_N_X, MAX_N_Y, MAX_N_Z, dims, comm_cart);

  init_conditions_heat3d(A, B);
  init_conditions_heat3d(gather);

  A.sweep_setup_heat3d(1, 1);
  B.sweep_setup_heat3d(1, 1);

  t1 = MPI_Wtime();
  for ( i = 0; i < MAX_it; ++i )
  {
    A.sweep_heat3d(B);
    A.I_exchange3d();

    B.sweep_heat3d(A);
    B.I_exchange3d();

    loc_diff = final_project::get_difference(A, B);
    MPI_Allreduce(&loc_diff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM, comm_cart);

    if (glob_diff <= tol) {break;}
  }
  t2 = MPI_Wtime();



  t2 -= t1;
  t1 = 0;
  MPI_Reduce(&t2, &t1, 1, MPI_DOUBLE, MPI_MAX, root, comm_cart);

  A.Gather3d(gather, root, comm_cart);
  if (world.rank() == root ) 
  {
    std::cout << "it" << " " << "t" << std::endl;
    std::cout << i << " " << t1 * 1000 << std::endl;
    std::cout << gather << std::endl;
    gather.saveToBinaryFile("mat.bin");
  }

  return 0;
}