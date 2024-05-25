#include <mpi.h>
#include "environment.cpp"
#include "array.cpp"
#include "sweep.cpp"
#include "exchange.cpp"

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

  final_project::array2d<double> A(12, 12);
  A.fill(-5);
  // std::cout << A << std::endl;

  final_project::array2d_distribute<double> a, b;
  a.distribute(10, 10, dims, comm_cart);
  b.distribute(10, 10, dims, comm_cart);

  a.fill(world.rank()+1);
  b.fill(world.rank()+2);

  a.sweep_setup_heat2d(1, 1);
  b.sweep_setup_heat2d(1, 1);

  // if (world.rank() == 1)
  //   std::cout << a << " \n " << a.starts[0] << ", " << a.ends[0] << std::endl;

  // std::cout << a.starts[0] << ", " << a.ends[0] << std::endl;    

  for ( int i = 0; i < 10; ++i )
  {
    a.sweep_heat2d(b);
    b.sweep_heat2d(a);
  }


  a.Gather(A, 0, comm_cart);
  if (world.rank() == 0)
    std::cout << A;
  

  return 0;
}