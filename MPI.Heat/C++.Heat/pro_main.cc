#if !defined(USE_MPI)
#include <mpi.h>
#include <omp.h>
// #include <boost/mpi.hpp>
// #include <boost/regex.hpp>
// #include <boost/serialization/vector.hpp>
// #include <boost/serialization/access.hpp>
#endif

#include <iostream>
#include <chrono>

#include <iomanip>
#include <memory>
#include <cstring>
#include <vector>
#include <algorithm>

#include "final_project.cpp"

#if !defined(MAX_N_X) || !defined(MAX_N_Y)
#define MAX_N_X 100+2
#define MAX_N_Y 100+2
#endif

#if !defined (MAX_it)
#define MAX_it 10000
#endif

int main(int argc, char ** argv)
{
  // boost::mpi::environment env(argc, argv);
  // boost::mpi::communicator world;
  auto world = mpi::env(argc, argv);

  constexpr int reorder {1}, dimension {2}, root {0};

  int dims[dimension], coords[dimension], periods[dimension];
  for (short int i = 0; i < dimension; ++i) {dims[i] = 0; periods[i] = 0;}

  /* MPI Cartesian Inits */
  MPI_Comm comm_cart;
  MPI_Dims_create(world.size(), dimension, dims);
  MPI_Cart_create(MPI_COMM_WORLD, dimension, dims, periods, reorder, &comm_cart);

  final_project::Array<double>        gather (MAX_N_X, MAX_N_Y, -5);
  final_project::Array_Distribute<double>  a (MAX_N_X, MAX_N_Y, dims, comm_cart), 
                            b (MAX_N_X, MAX_N_Y, dims, comm_cart), 
                            f (MAX_N_X, MAX_N_Y, dims, comm_cart);

  twodinit_basic_Heat(a, b, f);

  auto t1 = MPI_Wtime();
  for (int i = 0; i < MAX_it; ++i)
  {

    a.sweep(b);
    b.Iexchange();

    b.sweep(a);
    a.Iexchange();

  }
  auto t2 = MPI_Wtime();

  std::cout 
  << "RANK: "      << world.rank()     << "\n"
  << "Parallel : " << (t2 - t1) * 1000 << " ms\n" 
  << std::endl;

  twodinit_basic_Heat(gather);
  a.Array_Gather(gather, 0);

  if (world.rank() == root ) std::cout << gather;

  return EXIT_SUCCESS;
}
