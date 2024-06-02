/**
 * @file main_1d.cc
 * 
 * @brief Main file for the parallel Heat 1d Equation solver using MPI.
 * 
 * This file sets up the MPI environment, initializes the arrays, performs
 * the heat equation computation, exchanges data between processes, gathers
 * the final results, and outputs the results.
 * 
 * @author Li Yihai
 * @version 3.0
 * @date Jun 2, 2024
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

#if !defined(MAX_N_X) || !defined(MAX_N_Y)
#define MAX_N 100+2
#endif

#if !defined (MAX_it)
#define MAX_it 12'000
#endif

#define tol 1E-13

int main(int argc, char ** argv)
{
  auto world = mpi::env(argc, argv);

  int i;
  double loc_diff, glob_diff {10}, t1, t2;
  constexpr int reorder {1}, dimension {1}, root {0};

  int dims[dimension], periods[dimension], corrds[dimension];
  for (short int i = 0; i < dimension; ++i) {dims[i] = 0; periods[i] = 0;}


  /* MPI Cartesian Inits */
  MPI_Comm comm_cart;
  MPI_Dims_create(world.size(), dimension, dims);
  MPI_Cart_create(MPI_COMM_WORLD, dimension, dims, periods, reorder, &comm_cart);

  final_project::array1d_distribute<double> a, b;
  final_project::array1d<double> gather (MAX_N);

  a.distribute(MAX_N, dims, comm_cart);
  b.distribute(MAX_N, dims, comm_cart);

  init_conditions_heat1d(a, b);
  init_conditions_heat1d(gather);

  a.sweep_setup_heat1d(1,1);
  b.sweep_setup_heat1d(1,1);

  t1 = MPI_Wtime();
  for (i = 0; i < MAX_it; ++i)
  {
    a.sweep_heat1d(b);
    b.I_exchange1d();

    b.sweep_heat1d(a);
    a.I_exchange1d();

    loc_diff = final_project::get_difference(a, b);
    MPI_Allreduce(&loc_diff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM, comm_cart);

    // if (world.rank() == root) std::cout << glob_diff << std::endl;
    if (glob_diff <= tol) {break;}

    if (i % 100 == 0) {
      char buffer[50];
      std::sprintf(buffer, "mat_%d.bin", i);
      a.Gather1d(gather, root, comm_cart);
      if (world.rank() == 0) gather.saveToBinaryFile(buffer);
    }
  }
  t2 = MPI_Wtime();

  // final_project::print_in_order(a);

  t2 -= t1;
  t1 = 0;
  MPI_Reduce(&t2, &t1, 1, MPI_DOUBLE, MPI_MAX, root, comm_cart);

  a.Gather1d(gather, root, comm_cart);
  
  MPI_Barrier(comm_cart);
  if (world.rank() == root)
  {
    std::cout << "it" << " " << "t" << std::endl;
    std::cout << i << " " << t1 * 1000 << std::endl;
    // std::cout << gather << std::endl;
    gather.saveToBinaryFile("mat.bin");
  }



  


  return 0;
}