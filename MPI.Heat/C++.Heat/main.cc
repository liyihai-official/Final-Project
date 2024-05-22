#include <iostream>
#include <chrono>

#include <iomanip>
#include <memory>
#include <cstring>
#include <vector>
#include <algorithm>
// #include <gperftools/profiler.h>

#include <omp.h>

#include "final_project.cpp"


#if !defined(USE_MPI)
#include <mpi.h>
// #include <boost/mpi.hpp>
// #include <boost/regex.hpp>
// #include <boost/serialization/vector.hpp>
// #include <boost/serialization/access.hpp>
#endif

#if !defined(MAX_N_X) || !defined(MAX_N_Y)
#define MAX_N_X 100+2
#define MAX_N_Y 100+2
#endif

#if !defined (MAX_it)
#define MAX_it 100'000'000
#endif

#define tol 1E-13

int main(int argc, char ** argv)
{
  // boost::mpi::environment env(argc, argv);
  // boost::mpi::communicator world;

  auto world = mpi::env(argc, argv);

  double loc_diff, glob_diff {10}, t1, t2;
  int i;
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

/* ------------------------------- With OMP version ------------------------------- */
#ifdef USE_OMP
  t1 = MPI_Wtime();
  #pragma omp parallel private(i) num_threads(2)
  for (i = 0; i < MAX_it; ++i)
  {
    int p_id = omp_get_thread_num();
    a.sweep3(b, p_id);
    #pragma omp barrier

    #pragma omp single
    {
      b.Iexchange();
    }

    b.sweep3(a, p_id);
    #pragma omp barrier

    #pragma omp single
    {
      a.Iexchange();
      loc_diff = final_project::get_difference(a, b);
      MPI_Allreduce(&loc_diff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM,     comm_cart);
    }

    if (glob_diff <= tol) {break;}
  }
  t2 = MPI_Wtime();

  /** OMP ver 1.0 */
  // // ProfilerStart("main.prof");
  // t1 = MPI_Wtime();
  // #pragma omp parallel private(i) num_threads(2)
  // for (i = 0; i < MAX_it; ++i)
  // {
  //   int p_id = omp_get_thread_num();
    
  //   a.sweep(b, p_id);
  //   #pragma omp barrier

  //   #pragma omp single
  //   {
  //     b.Iexchange();
  //   }

  //   b.sweep(a, p_id);
  //   #pragma omp barrier

  //   #pragma omp single
  //   {
  //     a.Iexchange();
  //     loc_diff = final_project::get_difference(a, b);
  //     MPI_Allreduce(&loc_diff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM,     comm_cart);
  //   }

  //   if (glob_diff <= tol) {break;}
  // }
  // t2 = MPI_Wtime();
  // // ProfilerStop();

/* -------------------------------- No OMP version -------------------------------- */
#else

  // ProfilerStart("main.prof");
  t1 = MPI_Wtime();
  for (i = 0; i < MAX_it; ++i)
  { 
    a.sweep(b);
    b.Iexchange();

    b.sweep(a);
    a.Iexchange();

    loc_diff = final_project::get_difference(a, b);
    MPI_Allreduce(&loc_diff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM,     comm_cart);

    if (glob_diff <= tol) {break;}
  }
  t2 = MPI_Wtime();
  // ProfilerStop();
  
#endif

  
  MPI_Barrier(comm_cart);

  double t_t {t2-t1};
  double t_list[world.size()];
  MPI_Gather(&t_t, 1, MPI_DOUBLE, t_list, 1, MPI_DOUBLE, root, comm_cart);

  
  if (world.rank() == root)
  {
    std::cout 
      << "num_proc"           << " "
      << "NX"                 << " "
      << "NY"                 << " "
      << "MAX_iteration"      << " "
      << "Convergence"        << " "
      << "Rank"               << " "
      << "Time"               << 
    std::endl;

    for (int j = 0; j < world.size(); ++j)
      std::cout 
        << world.size()     << " "
        << MAX_N_X          << " "
        << MAX_N_Y          << " "
        << MAX_it           << " "
        << i                << " "
        << j                << " "
        << t_list[j] * 1000 << 
      std::endl;
  }

  twodinit_basic_Heat(gather);
  a.Array_Gather(gather, 0);

  // if (world.rank() == root ) std::cout << gather;

  return EXIT_SUCCESS;
}
