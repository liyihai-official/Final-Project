#define USE_MPI

#ifdef USE_MPI
#include <mpi.h>
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

#define MAX_N 1000+2
#define MAX_it 1000000000

template<typename T>
MPI_Datatype get_mpi_type();

template<>
MPI_Datatype get_mpi_type<int>() { return MPI_INT; }

template<>
MPI_Datatype get_mpi_type<float>() { return MPI_FLOAT; }

template<>
MPI_Datatype get_mpi_type<double>() { return MPI_DOUBLE; }

template <typename T>
class Array_Distribute : public Array<T> {
  Array_Distribute() = delete;

  Array_Distribute(std::size_t const rows, std::size_t const cols, 
                    int const dims[2], int const coords[2], MPI_Comm comm_cart)
  {
    Array<T>(rows, cols);

    nx = rows - 2;
    ny = cols - 2;

    Decomp1d(nx, dims[0], coords[0], starts[0], ends[0]);
    Decomp1d(ny, dims[1], coords[1], starts[1], ends[1]);

    MPI_Cart_shift(comm_cart, 0, 1, &nbr_up,   &nbr_down );
    MPI_Cart_shift(comm_cart, 1, 1, &nbr_left, &nbr_right);

    MPI_Comm_rank(comm_cart, &rank);

    /* Setup vector types of halo */
    MPI_Type_contiguous(ends[1] - starts[1] + 1, get_mpi_type<T>(), &vecs[0]);
    MPI_Type_commit(&vecs[0]);

    MPI_Type_vector(ends[0] - starts[0] + 1, 1, MAX_N, get_mpi_type<T>(), &vecs[1]);
    MPI_Type_commit(&vecs[1]);

  }
  

  protected:
  constexpr static int dimension {2};

  private:
  const int nx, ny;

  const std::size_t starts[2], ends[2];

  int rank;
  int nbr_up, nbr_down, nbr_right, nbr_left;

  MPI_Comm comm;

  MPI_Datatype vecs[dimension];

};

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
