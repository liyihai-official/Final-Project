/**
 * @file main_omp1.cc
 * 
 * @brief 1 first try of Main file for the parallel Heat Equation solver 
 * using MPI and OpenMP.
 * 
 * This file sets up the MPI environment, initializes the arrays, performs
 * the heat equation computation, exchanges data between processes, gathers
 * the final results, and outputs the results. Moreover, this method calls 
 * OpenMP functions while iterating, and doing even-odd sweeping strategy aiming
 * for boosting up the updating processes.
 * 
 * @author Li Yihai
 * @version 3.1
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
#include <omp.h>

#include "final_project.cpp"

#if !defined(MAX_N_X) || !defined(MAX_N_Y)
#define MAX_N_X 100+2
#define MAX_N_Y 100+2
#endif

#if !defined (MAX_it)
#define MAX_it 100'000'000
#endif

#define tol 1E-13

int main (int argc, char ** argv)
{
  auto world = mpi::env(argc, argv);

  int iteration {0};
  double loc_diff, glob_diff {10}, t1, t2;
  constexpr int reorder {1}, dimension {2}, root {0};

  int dims[dimension], coords[dimension], periods[dimension];
  for (short int i = 0; i < dimension; ++i) {dims[i] = 0; periods[i] = 0;}

  /* MPI Cartesian Inits */
  MPI_Comm comm_cart;
  MPI_Dims_create(world.size(), dimension, dims);
  MPI_Cart_create(MPI_COMM_WORLD, dimension, dims, periods, reorder, &comm_cart);

  final_project::array2d_distribute<double> a, b;
  final_project::array2d<double> gather (MAX_N_X, MAX_N_Y);

  a.distribute(MAX_N_X, MAX_N_Y, dims, comm_cart);
  b.distribute(MAX_N_X, MAX_N_Y, dims, comm_cart);

  init_conditions_heat2d(a, b);
  init_conditions_heat2d(gather);

  a.sweep_setup_heat2d(1, 1);
  b.sweep_setup_heat2d(1, 1);

  t1 = MPI_Wtime();
#pragma omp parallel num_threads(2)
  for (int i = 0; i < MAX_it; ++i )
  {
    int p_id = omp_get_thread_num();
    
    a.sweep_heat2d_omp1(b, p_id);
#pragma omp barrier

#pragma omp single
    {
      b.I_exchange2d();
    }

#pragma omp barrier
    b.sweep_heat2d_omp1(a, p_id);
#pragma omp barrier

#pragma omp single
    {
      a.I_exchange2d();

      loc_diff = final_project::get_difference(a, b);
      MPI_Allreduce(&loc_diff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM, comm_cart);
    }

#pragma omp barrier
    if (glob_diff <= tol) {
      iteration = i;
      break;
    }
#pragma omp barrier

  }
  t2 = MPI_Wtime();
  
  // final_project::print_in_order(a);

  t2 -= t1;
  t1 = 0;
  MPI_Reduce(&t2, &t1, 1, MPI_DOUBLE, MPI_MAX, root, comm_cart);

  a.Gather2d(gather, root, comm_cart);
  if (world.rank() == root)
  {
    std::cout << "it" << " " << "t" << std::endl;
    std::cout <<  iteration << " " << t1 * 1000 << std::endl;
    // std::cout << gather << std::endl;
  }
    
  return 0;
}