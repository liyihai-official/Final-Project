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

#define MAX_N 12+2
#define MAX_it 10000

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
  public:
  Array_Distribute() = delete;

  Array_Distribute(std::size_t const rows, std::size_t const cols, 
                          int const dims[2], MPI_Comm comm_cart)
    : Array<T>( get_loc_dim(rows - 2, dims, 0, comm_cart), 
                get_loc_dim(cols - 2, dims, 1, comm_cart)),
      nx_glob {static_cast<int>(rows - 2)}, 
      ny_glob {static_cast<int>(cols - 2)}, 
      comm {comm_cart}
  {

    nx_loc = ends[0] - starts[0] + 1;
    ny_loc = ends[1] - starts[1] + 1;

    MPI_Cart_shift(comm_cart, 0, 1, &nbr_up,   &nbr_down );
    MPI_Cart_shift(comm_cart, 1, 1, &nbr_left, &nbr_right);

    /* Setup vector types of halo */
    MPI_Type_contiguous(ends[1] - starts[1] + 1, get_mpi_type<T>(), &vecs[0]);
    MPI_Type_commit(&vecs[0]);

    MPI_Type_vector(ends[0] - starts[0] + 1, 1, MAX_N, get_mpi_type<T>(), &vecs[1]);
    MPI_Type_commit(&vecs[1]);

    std::cout << "Process " << rank 
    << " - (" << coordinates[0] << ", " << coordinates[1] << ")\t"
    << starts[0] << ", " << ends[0] << " - " << starts[1] << ", " << ends[1] 
    << std::endl;

  }
  

  int get_start(const int idx) {
    if (idx == 0 || idx == 1) 
      return starts[idx];
    else 
    {
      std::string msg {"Array subscript out of index.\n"};
      throw std::out_of_range(msg);
    }
  }

  int get_end(const int idx) {
    if (idx == 0 || idx == 1) 
      return ends[idx];
    else 
    {
      std::string msg {"Array subscript out of index.\n"};
      throw std::out_of_range(msg);
    }
  }

  protected:
  constexpr static int dimension {2};







  private:
  int rank;
  int nx_loc, ny_loc, nx_glob, ny_glob;
  int nbr_up, nbr_down, nbr_right, nbr_left;

  int starts[dimension], ends[dimension], coordinates[dimension];

  MPI_Comm comm;
  MPI_Datatype vecs[dimension];

  std::size_t get_loc_dim(auto glob_dim, const int dims[], const int coord_idx, MPI_Comm comm)
  {
    int start, end;
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, dimension, coordinates);

    // std::cout << dims[coord_idx] << "\t" << coordinates[coord_idx] << std::endl;
    Decomp1d(glob_dim, dims[coord_idx], coordinates[coord_idx], starts[coord_idx], ends[coord_idx]);
    
    return ends[coord_idx] - starts[coord_idx] + 1;
  };

};

template <typename T>
void twodinit_basic_Heat(Array_Distribute<T>& init, Array_Distribute<T>& init_other, Array_Distribute<T> bias)
{
  int i, j;
  int nx = init.get_num_rows() - 2;
  int ny = init.get_num_cols() - 2;

  int s[] {init.get_start(0), init.get_start(1)};
  int e[] {init.get_end(0),   init.get_end(1)};

  for (i = s[0]-1; i <= e[0]+1; ++i) 
    for (j = s[1]-1; j <= e[1]+1; ++j)
    {
      init(i, j) = 0;
      init_other(i, j) = 0;
      bias(i, j) = 0;
    }

  /* Left Side */
  if (s[0] == 1) {
    for (j = s[1]; j < e[1]+1; ++j) {
      double yy = (double) j / (ny+1);
      init(0,       j) = 10;
      init_other(0, j) = 10;
    }
  }

  /* Right Side */
  if (e[0] == nx) {
    for (j = s[1]; j < e[1]+1; ++j) {
      double yy = (double) j / (ny+1);
      init(nx+1,        j) = 10;
      init_other(nx+1,  j) = 10;
    }
  }

  /* Bottom side */
  if (s[1] == 1) {
    for (i = s[0]; i <= e[0]; ++i) {
      init(i,       0) = 10;
      init_other(i, 0) = 10;
    }
  }
  
  /* UP side */
  if (e[1] == ny) {
    for (i = s[0]; i <= e[0]; ++i) {
      double xx = (double) i / (nx+1);
      init(i,       ny+1) = 0;
      init_other(i, ny+1) = 0;
    }
  }

}



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

  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  constexpr int reorder {1}, dimension {2};

  int dims[dimension], coords[dimension], periods[dimension];
  for (short int i = 0; i < dimension; ++i) {dims[i] = 0; periods[i] = 0;}

  /* MPI Cartesian Inits */
  MPI_Comm comm_cart;
  MPI_Dims_create(world.size(), dimension, dims);
  MPI_Cart_create(MPI_COMM_WORLD, dimension, dims, periods, reorder, &comm_cart);

  Array_Distribute<double>  a (MAX_N, MAX_N, dims, comm_cart), 
                            b (MAX_N, MAX_N, dims, comm_cart), 
                            f (MAX_N, MAX_N, dims, comm_cart);

  if (world.rank() == 0)
    std::cout << a << std::endl; 
  std::cout << std::endl;   

  twodinit_basic_Heat(a, b, f);

  std::cout << std::endl;   
  if (world.rank() == 0)
    std::cout << a << std::endl;   


















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
