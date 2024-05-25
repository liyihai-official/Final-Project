#include <mpi.h>
#include "environment.cpp"
#include "array.cpp"
#include "sweep.cpp"
#include "exchange.cpp"
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
  final_project::array2d<double> gather (12, 12);
  final_project::array2d_distribute<double> a, b;
  a.distribute(10, 10, dims, comm_cart);
  b.distribute(10, 10, dims, comm_cart);

  init_conditions_heat2d(a, b);
  init_conditions_heat2d(gather);

  a.sweep_setup_heat2d(1, 1);
  b.sweep_setup_heat2d(1, 1);

  for ( int i = 0; i < 10; ++i )
  {
    a.sweep_heat2d(b);
    b.sweep_heat2d(a);

    loc_diff = final_project::get_difference(a, b);
    MPI_Allreduce(&loc_diff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM, comm_cart);

    if (glob_diff <= tol) {break;}
  }

  a.Gather2d(gather, 0, comm_cart);
  if (world.rank() == 0)
  {
    std::cout << a;
  //   std::cout << "a :" << a.rows() << ", "  << a.cols() << "\n" <<
  //   "b :" << b.rows() << ", "  << b.cols() << "\n";
  //   std::cout << final_project::get_difference(a, b) << std::endl;
  }
    
    
    
  

  return 0;
}