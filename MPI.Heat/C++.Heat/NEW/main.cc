/**
 * @file main.cpp
 * 
 * @brief Main file for the parallel Heat Equation solver using MPI.
 * 
 * This file sets up the MPI environment, initializes the arrays, performs
 * the heat equation computation, exchanges data between processes, gathers
 * the final results, and outputs the results.
 * 
 * @author Li Yihai
 * @version 3.0
 * @date May 25, 2024
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

#define tol 1E-13


int main (int argc, char ** argv)
{
  auto world = mpi::env(argc, argv);

  int i;
  double loc_diff, glob_diff {10}, t1, t2;
  constexpr int reorder {1}, dimension {2}, root {0};

  int dims[dimension], coords[dimension], periods[dimension];
  for (short int i = 0; i < dimension; ++i) {dims[i] = 0; periods[i] = 0;}

  /* MPI Cartesian Inits */
  MPI_Comm comm_cart;
  MPI_Dims_create(world.size(), dimension, dims);
  MPI_Cart_create(MPI_COMM_WORLD, dimension, dims, periods, reorder, &comm_cart);

  final_project::array2d_distribute<double> a, b;
  final_project::array2d<double> gather (12, 12);

  a.distribute(12, 12, dims, comm_cart);
  b.distribute(12, 12, dims, comm_cart);

  init_conditions_heat2d(a, b);
  init_conditions_heat2d(gather);

  a.sweep_setup_heat2d(1, 1);
  b.sweep_setup_heat2d(1, 1);

  t1 = MPI_Wtime();
  for ( i = 0; i < 1000000; ++i )
  {
    a.sweep_heat2d(b);
    b.I_exchange2d();

    b.sweep_heat2d(a);
    a.I_exchange2d();

    loc_diff = final_project::get_difference(a, b);
    MPI_Allreduce(&loc_diff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM, comm_cart);

    if (glob_diff <= tol) {break;}
  }
  t2 = MPI_Wtime();
  
  // final_project::print_in_order(a);

  t2 -= t1;
  t1 = 0;
  MPI_Reduce(&t2, &t1, 1, MPI_DOUBLE, MPI_MAX, root, comm_cart);

  a.Gather2d(gather, root, comm_cart);
  if (world.rank() == root)
  {
    std::cout << i << " " << t1 * 1000 << std::endl;
    std::cout << gather << std::endl;
  }
   
    
  return 0;
}