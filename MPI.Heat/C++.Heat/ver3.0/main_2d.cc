/**
 * @file main_2d.cc
 * 
 * @brief Main file for the parallel Heat 2d Equation solver using MPI.
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
#include "heat.cpp"

#if !defined(MAX_N_X) || !defined(MAX_N_Y)
#define MAX_N_X 200+2
#define MAX_N_Y 200+2
#endif

#if !defined (MAX_it)
#define MAX_it 2'000'000
#endif

#define tol 1E-10

int main (int argc, char ** argv)
{
  auto world = mpi::env(argc, argv);

  int i;
  double loc_diff, glob_diff {10}, t1, t2;
  constexpr int reorder {1}, dimension {2}, root {0};

  int dims[dimension], periods[dimension];
  for (short int i = 0; i < dimension; ++i) {dims[i] = 0; periods[i] = 0;}

  /* MPI Cartesian Inits */
  MPI_Comm comm_cart;
  MPI_Dims_create(world.size(), dimension, dims);
  MPI_Cart_create(MPI_COMM_WORLD, dimension, dims, periods, reorder, &comm_cart);

  auto gather = final_project::array2d<double>(MAX_N_X, MAX_N_Y);
  gather.fill(0);

  auto a = final_project::heat_equation::heat2d_pure_mpi<double>(MAX_N_X, MAX_N_Y, dims, comm_cart);
  auto b = final_project::heat_equation::heat2d_pure_mpi<double>(MAX_N_X, MAX_N_Y, dims, comm_cart);  

  init_conditions_heat2d(a.body, b.body);


  t1 = MPI_Wtime();
  for ( i = 0; i < MAX_it; ++i )
  {
    a.sweep_heat2d(b);
    b.body.I_exchange2d();

    b.sweep_heat2d(a);
    a.body.I_exchange2d();

    loc_diff = final_project::get_difference(a.body, b.body);
    MPI_Allreduce(&loc_diff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM, comm_cart);

    if (glob_diff <= tol) {break;}

    // if (i % 1000 == 0) {
    //   char buffer[50];
    //   std::sprintf(buffer, "visualize/mat_%d.bin", i);
    //   a.Gather2d(gather, root, comm_cart);
    //   if (world.rank() == 0) gather.saveToBinaryFile(buffer);
    // }

  }
  t2 = MPI_Wtime();
  
  final_project::print_in_order(a.body);

  t2 -= t1;
  t1 = 0;
  MPI_Reduce(&t2, &t1, 1, MPI_DOUBLE, MPI_MAX, root, comm_cart);

  
  a.body.Gather2d(gather, root, comm_cart);
  if (world.rank() == root)
  {
    std::cout << "it" << " " << "t" << std::endl;
    std::cout << i << " " << t1 * 1000 << std::endl;
    std::cout << gather << std::endl;
    // gather.saveToBinaryFile("mat.bin");
  }
   
  return 0;
}