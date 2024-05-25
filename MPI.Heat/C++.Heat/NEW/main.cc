
#include <mpi.h>
#include "environment.cpp"

#include "array.cpp"
#include "sweep.cpp"
#include "exchange.cpp"
#include "gather.cpp"

#include "initialization.cpp"

#define tol 1E-13


int main (int argc, char ** argv)
{
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

  /****************************************************************************************/
  final_project::array2d<double> gather (22, 22);
  final_project::array2d_distribute<double> a, b;
  a.distribute(22, 22, dims, comm_cart);
  b.distribute(22, 22, dims, comm_cart);

  init_conditions_heat2d(a, b);
  init_conditions_heat2d(gather);

  a.sweep_setup_heat2d(1, 1);
  b.sweep_setup_heat2d(1, 1);

  t1 = MPI_Wtime();
  for ( i = 0; i < 10000; ++i )
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
  
  final_project::print_in_order(a);
  // MPI_Barrier(comm_cart);

  // double t_t {t2-t1};
  // double t_list[world.size()];
  // MPI_Gather(&t_t, 1, MPI_DOUBLE, t_list, 1, MPI_DOUBLE, root, comm_cart);

  // if (world.rank() == root)
  // {
  //   std::cout 
  //     << "num_proc"           << " "
  //     << "Convergence"        << " "
  //     << "Rank"               << " "
  //     << "Time"               << 
  //   std::endl;

  //   for (int j = 0; j < world.size(); ++j)
  //     std::cout 
  //       << world.size()     << " "
  //       << i                << " "
  //       << j                << " "
  //       << t_list[j] * 1000 << 
  //     std::endl;
  // }
  // a.Gather2d(gather, 0, comm_cart);
  // if (world.rank() == 0)
  // {
    // std::cout << gather;
  //   std::cout << "a :" << a.rows() << ", "  << a.cols() << "\n" <<
  //   "b :" << b.rows() << ", "  << b.cols() << "\n";
  //   std::cout << final_project::get_difference(a, b) << std::endl;
  // }
    
    
    
  return 0;
}