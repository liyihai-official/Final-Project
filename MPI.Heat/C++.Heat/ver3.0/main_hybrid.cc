#include "multi_array/array_hybrid.cpp"

#include <mpi.h>

#include "final_project.cpp"
#include "heat.cpp"
#include <omp.h>


#if !defined(MAX_N_X) || !defined(MAX_N_Y)
#define MAX_N_X 200+2
#define MAX_N_Y 200+2
#endif

#if !defined (MAX_it)
#define MAX_it 2'000'000
#endif

#define tol 1E-10


template <class T>
void Sweep(final_project::array2d_hybrid<T> & in)
{
  const int omp_id {omp_get_thread_num()};
  int off;

  int x, y;
  for (x = 1; x <= in.rows() - 2; ++ x)
  { 
    off = 1 + ( x + omp_id + 1) % 2;
    for (y = off ; y <= in.cols() - 2; y+=2)
    {
      in(x,y) = 0.25 * (in(x+1, y) + in(x-1, y) + 
                        in(x, y+1) + in(x, y-1));
    }
  }
}

template <class T>
void SweepSS(final_project::array2d_hybrid<T> & in, final_project::array2d_hybrid<T> & out)
{
  int x, y;
  for (x = 1; x <= in.rows() - 2; ++ x)
  { 
    for ( y = 1 ; y <= in.cols() - 2; ++ y)
    {
      out(x,y) = 0.25 * (in(x+1, y) + in(x-1, y) + 
                         in(x, y+1) + in(x, y-1));
    }
  }
}


int main ( int argc, char ** argv)
{
  auto world {mpi::env(argc, argv)};

  constexpr int reorder {1}, dimension {2}, root {0};

  int dims[dimension], periods[dimension];
  for (short int i = 0; i < dimension; ++i) {dims[i] = 0; periods[i] = 0;}

  MPI_Comm comm_cart;
  MPI_Dims_create(world.size(), dimension, dims);
  MPI_Cart_create(MPI_COMM_WORLD, dimension, dims, periods, reorder, &comm_cart);

  auto gather = final_project::array2d<double>(MAX_N_X, MAX_N_Y);
  gather.fill(-1);

  auto a = final_project::heat_equation::heat2d_pure_mpi<double>(MAX_N_X, MAX_N_Y, dims, comm_cart);
  auto b = final_project::heat_equation::heat2d_pure_mpi<double>(MAX_N_X, MAX_N_Y, dims, comm_cart);  

  init_conditions_heat2d(a.body, b.body);

  final_project::array2d_hybrid<double> H, G;
  H.mpi_distribute(MAX_N_X, MAX_N_Y, dims, comm_cart);
  G.mpi_distribute(MAX_N_X, MAX_N_Y, dims, comm_cart);


  ////
  H.fill(0);
  G.fill(0);
  if (H.starts[0] == 1) for (int j = 0; j < H.cols(); ++j) H(0,j) = 10;
  if (H.starts[1] == 1) for (int i = 0; i < H.rows(); ++i) H(i,0) = 10;

  if (H.ends[0] == H.global_Rows-2) for (int j = 0; j < H.cols(); ++j) H(H.rows()-1,j) = 10;
  if (H.ends[1] == H.global_Cols-2) for (int i = 0; i < H.rows(); ++i) H(i,H.cols()-1) = 10;

  
  if (G.starts[0] == 1) for (int j = 0; j < G.cols(); ++j) G(0,j) = 10;
  if (G.starts[1] == 1) for (int i = 0; i < G.rows(); ++i) G(i,0) = 10;

  if (G.ends[0] == G.global_Rows-2) for (int j = 0; j < G.cols(); ++j) G(G.rows()-1,j) = 10;
  if (G.ends[1] == G.global_Cols-2) for (int i = 0; i < G.rows(); ++i) G(i,G.cols()-1) = 10;
  ////



#pragma omp parallel num_threads(2)
{
  MPI_Datatype halos[dimension];
  H.hybrid_halo(halos, omp_get_thread_num());
  auto t1 = MPI_Wtime();
  // if (omp_get_thread_num() == 0)
  {
  for (int i = 0; i < 1; ++i)
  {
    Sweep(H);
    // Sweep(G, H);
  }
  }
  auto t2 = MPI_Wtime();
}
  final_project::print_in_order(H);


// auto t1 = MPI_Wtime();
// for ( int i = 0; i < 1 ; ++i)
// {
//   SweepSS(H, G);
//   SweepSS(G, H);
// }
// auto t2 = MPI_Wtime();

  // t2 -= t1;
  // t1 = 0;
  // MPI_Reduce(&t2, &t1, 1, MPI_DOUBLE, MPI_MAX, root, comm_cart);
  

  a.body.Gather2d(gather, root, comm_cart);
  if (world.rank() == root)
  {
    // std::cout << "it" << " " << "t" << std::endl;
    // std::cout << 0 << " " << t1 * 1000 << std::endl;
    // std::cout << gather << std::endl;
    // gather.saveToBinaryFile("mat.bin");
  }

  




  return 0;
}