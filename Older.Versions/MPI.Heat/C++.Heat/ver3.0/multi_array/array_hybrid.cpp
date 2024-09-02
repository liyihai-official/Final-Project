///
/// @file array_hybrid.cpp
///
///
#ifndef FINAL_PROJECT_ARRAY_HYBRID_CPP
#define FINAL_PROJECT_ARRAY_HYBRID_CPP

#pragma once
#include "../types.cpp"
#include "../assert.cpp"
#include "base.cpp"
#include <omp.h>

namespace final_project {

template <class T>
class array2d_hybrid : public array2d<T> {
  public:
  using typename array2d<T>::size_type;

  public:
  size_type global_Rows, global_Cols;

  // MPI_Topology Features
  int rank, num_proc;
  int nbr_up, nbr_down, nbr_right, nbr_left;

  constexpr static int dimension {2};
  int starts[dimension], ends[dimension], coordinates[dimension];

  MPI_Comm     communicator;

  public:
  array2d_hybrid() : global_Rows {0}, global_Cols {0}, array2d<T>(0, 0) {};

  public:
  void mpi_distribute(size_type gRows, size_type gCols, const int dims[2], MPI_Comm comm)
  {
    communicator = comm;
    global_Rows = gRows; global_Cols = gCols;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_proc);

    MPI_Cart_coords(comm, rank, dimension, coordinates);

    Decomp1d(global_Rows-2, dims[0], coordinates[0], starts[0], ends[0]);
    Decomp1d(global_Cols-2, dims[1], coordinates[1], starts[1], ends[1]);

    MPI_Cart_shift(comm, 0, 1, &nbr_up,   &nbr_down );
    MPI_Cart_shift(comm, 1, 1, &nbr_left, &nbr_right);

    this->resize(ends[0] - starts[0] + 3, ends[1] - starts[1] + 3);
  }


  void hybrid_halo(MPI_Datatype halos [2], const int omp_id)
  {
    const int nx {ends[0] - starts[0] + 1};
    const int ny {ends[1] - starts[1] + 1};

    const int nx_cnt {(omp_id == 0) ? ( nx / 2 + 1 ) : ( nx - nx / 2 )};
    const int ny_cnt {(omp_id == 0) ? ( ny / 2 + 1 ) : ( ny - ny / 2 )};
    
    MPI_Type_vector(ny_cnt, 1, 2,             get_mpi_type<T>(), &halos[0]);
    MPI_Type_vector(nx_cnt, 1, 2 * ( ny + 2), get_mpi_type<T>(), &halos[1]);

    MPI_Type_commit(&halos[0]);
    MPI_Type_commit(&halos[1]);
  }

  public:
  void SR_exchange2d();  
}; // class array2d_hybrid


template <class T>
void array2d_hybrid<T>::SR_exchange2d()
{

}

// template <class T>
// void array2d_hybrid<T>::Sweep2d(array2d_hybrid<T> out)
// {
//   const int omp_id {omp_get_thread_num()};
//   int off;

//   int x, y;
//   for (x = 1; x < this->rows() - 2; ++ x)
//   { 
//     off = 1 + ( x + omp_id + 1) % 2;
//     for ( y = off ; y < this->cols() - 2; ++y)
//     {
//       out(x,y) = weight
//     }
//   }
// }



} // namespace final_project




#endif // end define FINAL_PROJECT_ARRAY_HYBRID_CPP
