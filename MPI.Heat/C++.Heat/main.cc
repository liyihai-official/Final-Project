#define USE_MPI

#ifdef USE_MPI
#include <mpi.h>
#include <omp.h>
#include <boost/mpi.hpp>
#include <boost/regex.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/access.hpp>
#endif

#include <iostream>
#include <chrono>

#include <iomanip>
#include <memory>
#include <vector>
#include <algorithm>

#include "array.h"
#include "lib2d.h"
#include "array_mpi.h"

#define MAX_N 50+2
#define MAX_it 10000

template<typename T>
MPI_Datatype get_mpi_type();

template <typename T>
void twoexchange(Array<T>& in, const int s[2], const int e[2], 
                          MPI_Datatype vecs[2], MPI_Win win, 
                          const int nbr_down, const int nbr_right);

template <typename T>
int my_Gather2d_new(Array<T>& gather, Array<T> a, 
                          const int s[2], const int e[2], const int root, 
                          MPI_Comm comm);

int main(int argc, char ** argv)
{
  Array<double> a(MAX_N, MAX_N), b(MAX_N, MAX_N), f(MAX_N, MAX_N), solution(MAX_N, MAX_N), gather(MAX_N, MAX_N);

  int nx, ny, it;
  nx = (MAX_N) - 2; 
  ny = (MAX_N) - 2; 
  double loc_diff = 0.0, glob_diff = 0.0, loc_err, glob_err, tol = 1E-15;

  /* ------------------------------------ Parallel Ver. ------------------------------------------ */
  constexpr int root=0;
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  /* MPI Topology */
  MPI_Win win_a, win_b;

  constexpr int reorder = 1, dimension = 2;

  int cart_rank;
  int nbr_up, nbr_down, nbr_right, nbr_left;

  int s[dimension], e[dimension];
  int dims[dimension], periods[dimension], coords[dimension];
  for (short int i = 0; i < dimension; ++i) {dims[i] = 0; periods[i] = 0;}

  MPI_Comm comm_cart;  
  MPI_Datatype vecs[dimension];

  /* MPI Cartesian Inits */
  MPI_Dims_create(world.size(), dimension, dims);
  MPI_Cart_create(MPI_COMM_WORLD, dimension, dims, periods, reorder, &comm_cart);

  MPI_Comm_rank(comm_cart, &cart_rank);
  MPI_Cart_coords(comm_cart, cart_rank, dimension, coords);

  MPI_Cart_shift(comm_cart, 0, 1, &nbr_up,   &nbr_down );
  MPI_Cart_shift(comm_cart, 1, 1, &nbr_left, &nbr_right);

  Decomp1d(nx, dims[0], coords[0], s[0], e[0]);
  Decomp1d(ny, dims[1], coords[1], s[1], e[1]);

  #ifdef HEAT
  twodinit_basic_Heat(a, b, f, s, e);
  #endif

  #ifdef POSSION
  twodinit_basic_Possion(a, b, f, s, e);
  #endif

  #ifdef DEBUG
  printf("Process %d-(%d,%d) has neighbors - (%d, %d), (%d, %d)\n", world.rank(),
          coords[0], coords[1], s[0], e[0], s[1], e[1]);
  #endif

  /* Setup vector types of halo */
  MPI_Type_contiguous(e[1] - s[1] + 1, MPI_DOUBLE, &vecs[0]);
  MPI_Type_commit(&vecs[0]);

  MPI_Type_vector(e[0] - s[0] + 1, 1, MAX_N, MPI_DOUBLE, &vecs[1]);
  MPI_Type_commit(&vecs[1]);

  /* Open Window for RMA operations */
  MPI_Win_create(&a(s[0]-1, 0), (MAX_N)*(e[0]-s[0]+3) * sizeof(double), sizeof(double), 
                                MPI_INFO_NULL, comm_cart, &win_a);
  MPI_Win_create(&b(s[0]-1, 0), (MAX_N)*(e[0]-s[0]+3) * sizeof(double), sizeof(double),   
                                MPI_INFO_NULL, comm_cart, &win_b);
                                
  auto t1 = MPI_Wtime();
  {
    for (it = 0; it < MAX_it; ++it) 
    {
      twoexchange(a, s, e, vecs, win_a, nbr_down, nbr_right);

      #ifdef POSSION
      sweep_Possion(a, f, b, s, e);
      #endif

      #ifdef HEAT
      sweep_Heat(a, b, s, e);
      #endif

      twoexchange(b, s, e, vecs, win_b, nbr_down, nbr_right);
      
      #ifdef POSSION
      sweep_Possion(b, f, a, s, e);
      #endif

      #ifdef HEAT
      sweep_Heat(b, a, s, e);
      #endif

      loc_diff = get_difference(a, b, s, e);
      MPI_Allreduce(&loc_diff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM,     comm_cart);

      if (glob_diff <= tol) {break;}
    }
  }
  auto t2 = MPI_Wtime();

  if (world.rank() == root) {std::cout << "Convergence at " << it << " Iterations.\n";}

  if (world.rank() == root) init_conditions(gather, b, f);
  my_Gather2d_new(gather, a, s, e, root, MPI_COMM_WORLD);

  if (world.rank() == root) 
  {
    store_Array(gather, "output_par.dat");
    std::cout << "Parallel : " 
    << (t2 - t1) * 1000
    << " ms\n" << std::endl;
  }

  /* MPI Free */
  for (int i = 0; i < dimension; ++i) MPI_Type_free(&vecs[i]);
  MPI_Win_free(&win_a);
  MPI_Win_free(&win_b);

  return EXIT_SUCCESS;
}


template <typename T>
void twoexchange(Array<T>& in, const int s[2], const int e[2], 
  MPI_Datatype vecs[2], MPI_Win win, 
  const int nbr_down, const int nbr_right)
{

  MPI_Aint offset;
  constexpr int scnt = 1;

  MPI_Win_fence(0, win);

  /* Dim 1 */
  offset = (MAX_N) + e[1];
  MPI_Put(&in(s[0], e[1]  ), scnt, vecs[1], nbr_right, offset, 1, vecs[1], win);

  offset = (MAX_N) + e[1] + 1;
  MPI_Get(&in(s[0], e[1]+1), scnt, vecs[1], nbr_right, offset, 1, vecs[1], win);

  /* Dim 0 */
  offset = s[1];
  MPI_Put(&in(e[0]  , s[1]), scnt, vecs[0], nbr_down, offset, 1, vecs[0], win);

  offset = (MAX_N) + s[1];
  MPI_Get(&in(e[0]+1, s[1]), scnt, vecs[0], nbr_down, offset, 1, vecs[0], win);

  MPI_Win_fence(0, win);

}

template <typename T>
int my_Gather2d_new(Array<T>& gather, Array<T> a, 
                          const int s[2], const int e[2], const int root, 
                          MPI_Comm comm)
{
  int i, pid, tag;
  int rank, size;
  int count, block_length;
  MPI_Datatype block, temp, mpi_T;

  mpi_T = get_mpi_type<T>();

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  int s0_list[size], s1_list[size], counts[size], block_lengths[size];

  count = e[0] - s[0] + 1;
  block_length = e[1] - s[1] + 1;
  MPI_Type_vector(count, block_length, MAX_N, mpi_T, &block);
  MPI_Type_commit(&block);

  MPI_Gather(&s[0],         1, MPI_INT, s0_list,        1, MPI_INT, root, comm);
  MPI_Gather(&s[1],         1, MPI_INT, s1_list,        1, MPI_INT, root, comm);
  MPI_Gather(&count,        1, MPI_INT, counts,         1, MPI_INT, root, comm);
  MPI_Gather(&block_length, 1, MPI_INT, block_lengths,  1, MPI_INT, root, comm);

  if (rank != root)
  {
    tag = rank;
    MPI_Send(&a(s[0], s[1]), 1, block, root, tag, comm);
  }
  MPI_Type_free(&block);


  if (rank == root)
  {
    for (pid = 0; pid < size; ++pid) 
    {
      if (pid == root)
      {
        for (i = s[0]; i < e[0]+1; ++i)
        {
          memcpy(&gather(i, s[1]), &a(i, s[1]), block_lengths[pid] * sizeof(T));
        }
      }

      if (pid != root) 
      {
        tag = pid;
        MPI_Type_vector(counts[pid], block_lengths[pid], MAX_N, mpi_T, &temp);
        MPI_Type_commit(&temp);

        MPI_Recv(&gather(s0_list[pid], s1_list[pid]), 1, 
                  temp, pid, tag, comm, MPI_STATUS_IGNORE);

        MPI_Type_free(&temp);  
      }
    }
  }

  return MPI_SUCCESS;
}


template<>
MPI_Datatype get_mpi_type<int>() { return MPI_INT; }

template<>
MPI_Datatype get_mpi_type<float>() { return MPI_FLOAT; }

template<>
MPI_Datatype get_mpi_type<double>() { return MPI_DOUBLE; }
