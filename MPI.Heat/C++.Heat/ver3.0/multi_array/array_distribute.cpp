/**
 * @file array_distribute.cpp
 * @brief This file contains the definition of the array classes
 *        and features for parallel processing.
 * 
 * The classes provided enhanced array features into
 * distributed 1D, 2D and 3D array used in parallel processing 
 * environments.
 * 
 * @author LI Yihai
 * @version 3.0
 * @date May 25, 2024
 */
#ifndef FINAL_PROJECT_ARRAY_DISTRIBUTE_HPP_LIYIHAI
#define FINAL_PROJECT_ARRAY_DISTRIBUTE_HPP_LIYIHAI

#pragma once
#include <iostream>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <iterator>
#include <unistd.h>

#include <mpi.h>
#include <fstream>

#include "../assert.cpp"
#include "base.cpp"

namespace final_project 
{

  template <class T>
  class array1d_distribute : public array1d<T> {
    public:
      std::size_t glob_N;

      /* MPI Topology Features */
      int rank, num_proc;
      int nbr_left, nbr_right;

      constexpr static int dimension {1};
      int starts[dimension], ends[dimension], coordinates[dimension];

      MPI_Comm communicator;
      MPI_Datatype type {get_mpi_type<T>()};
    
    public:
      array1d_distribute() : glob_N {0}, array1d<T> (0) {};

    public:
      void distribute(std::size_t gN, const int dims[1], MPI_Comm comm)
      {
        communicator = comm;
        glob_N = gN;

        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &num_proc);
        MPI_Cart_coords(comm, rank, dimension, coordinates);

        Decomp1d(glob_N - 2, dims[0], coordinates[0], starts[0], ends[0]);

        MPI_Cart_shift(comm, 0, 1, &nbr_left, &nbr_right);

        const int nx {ends[0] - starts[0] + 1 + 2};

        this->resize(nx);
      }

      public:
        void I_exchange1d();
        void SR_exchange1d();

        void Gather1d(array1d<T>& gather, const int root, MPI_Comm comm);

  }; /* class array1d_distribute */


  /**
   * @brief A distributed 2D array class.
   * 
   * @tparam T The type of the elements stored in the array.
   */
  template <class T>
  class array2d_distribute : public array2d<T> {
    public:
      std::size_t glob_Rows, glob_Cols;

      /* MPI Topology Features */
      int rank, num_proc;
      int nbr_up, nbr_down, nbr_right, nbr_left;

      constexpr static int dimension {2};
      int starts[dimension], ends[dimension], coordinates[dimension];

      MPI_Comm     communicator;
      MPI_Datatype vecs[dimension];

    // Constructors
    public:
      array2d_distribute() : glob_Cols {0}, glob_Rows{0}, array2d<T>(0,0) {} ;

    public:
      /**
       * @brief Generate the distributed array by inputting communicator and global sizes.
       * 
       * @param gRows Global number of rows.
       * @param gCols Global number of columns.
       * @param dims Array of dimensions.
       * @param comm MPI communicator.
      */
      void distribute(std::size_t gRows, std::size_t gCols, const int dims[2], MPI_Comm comm)
      {
        communicator = comm;
        glob_Rows = gRows; glob_Cols = gCols;
        
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &num_proc);
        MPI_Cart_coords(comm, rank, dimension, coordinates);

        Decomp1d(glob_Rows-2, dims[0], coordinates[0], starts[0], ends[0]);
        Decomp1d(glob_Cols-2, dims[1], coordinates[1], starts[1], ends[1]);
        
        MPI_Cart_shift(comm, 0, 1, &nbr_up,   &nbr_down );
        MPI_Cart_shift(comm, 1, 1, &nbr_left, &nbr_right);

        /* Setup vector types of halo */ 
        const int nx {ends[0] - starts[0] + 1 + 2};
        const int ny {ends[1] - starts[1] + 1 + 2};

        MPI_Type_contiguous(ends[1] - starts[1] + 1,         get_mpi_type<T>(), &vecs[0]);
        MPI_Type_commit(&vecs[0]);

        MPI_Type_vector(    ends[0] - starts[0] + 1, 1, ny , get_mpi_type<T>(), &vecs[1]);
        MPI_Type_commit(&vecs[1]);
        
        this->resize(nx, ny);
      }


//       void distribute(std::size_t gRows, std::size_t gCols, 
//                       const int dims[2], const int omp_pid, 
//                       MPI_Datatype vecs_omp[2], MPI_Comm comm)
//       {

// FINAL_PROJECT_ASSERT_MSG((omp_pid == 0 || omp_pid == 1), "Invalid OpenMP even-odd setting. Require exactly 2 threads.");

//         communicator = comm;
//         glob_Rows = gRows; glob_Cols = gCols;
        
//         MPI_Comm_rank(comm, &rank);
//         MPI_Comm_size(comm, &num_proc);
//         MPI_Cart_coords(comm, rank, dimension, coordinates);

//         Decomp1d(glob_Rows-2, dims[0], coordinates[0], starts[0], ends[0]);
//         Decomp1d(glob_Cols-2, dims[1], coordinates[1], starts[1], ends[1]);
        
//         MPI_Cart_shift(comm, 0, 1, &nbr_up,   &nbr_down );
//         MPI_Cart_shift(comm, 1, 1, &nbr_left, &nbr_right);

//         /* Setup vector types of halo */ 
//         const int nx {ends[0] - starts[0] + 1};
//         const int ny {ends[1] - starts[1] + 1};

//         const int nx_cnt {(omp_pid == 0) ? (nx / 2 + 1) : (nx - nx / 2)};
//         const int ny_cnt {(omp_pid == 0) ? (ny / 2 + 1) : (ny - ny / 2)};

//     #pragma omp shared(glob_Rows, glob_Cols, rank, num_proc, nbr_up, nbr_down, nbr_right, nbr_left, communicator, starts, ends, coordinates) private(omp_pid)
//     {
//         MPI_Type_vector(ny_cnt, 1, 2, get_mpi_type<T>(), &vecs_omp[0]);
//         MPI_Type_commit(&vecs_omp[0]);

//         MPI_Type_vector(nx_cnt, 1, 2*(ny+2), get_mpi_type<T>(), &vecs_omp[1]);
//         MPI_Type_commit(&vecs_omp[1]);
//     }

//         this->resize(nx+2, ny+2);
//       }
    
    // exchanges, communications
    public:
      void I_exchange2d();
      void SR_exchange2d();

      void Gather2d(array2d<T>& gather, const int root, MPI_Comm comm);
  }; /* class array2d_distribute */


  template <class T>
  class array3d_distribute : public array3d<T> {
    public:
      std::size_t glob_Rows, glob_Cols, glob_Heights;

      /* MPI Topology Features */
      int rank, num_proc;
      int nbr_up, nbr_down, nbr_right, nbr_left, nbr_front, nbr_back;

      constexpr static int dimension {3};

      int starts[dimension], ends[dimension], coordinates[dimension];

      MPI_Datatype vecs[dimension];
      MPI_Comm     communicator;


    public: 
      array3d_distribute() : glob_Rows{0}, glob_Cols{0}, glob_Heights{0}, array3d<T>(0,0,0) {}; 

      /**
       * @brief Generate the distributed array by inputting communicator and global sizes.
       * 
       * @param gRows Global number of rows.
       * @param gCols Global number of columns.
       * @param dims Array of dimensions.
       * @param comm MPI communicator.
      */
      void distribute(std::size_t gRows, std::size_t gCols, std::size_t gHeights, 
                        const int dims[3], MPI_Comm comm)
      {
        communicator = comm;
        glob_Rows = gRows; glob_Cols = gCols, glob_Heights = gHeights;

        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &num_proc);
        MPI_Cart_coords(comm, rank, dimension, coordinates);

        MPI_Cart_shift(comm, 0, 1, &nbr_back , &nbr_front);
        MPI_Cart_shift(comm, 1, 1, &nbr_up   , &nbr_down );
        MPI_Cart_shift(comm, 2, 1, &nbr_left , &nbr_right);

        Decomp1d(glob_Rows-2, dims[0], coordinates[0], starts[0], ends[0]);
        Decomp1d(glob_Cols-2, dims[1], coordinates[1], starts[1], ends[1]);
        Decomp1d(glob_Heights-2, dims[2], coordinates[2], starts[2], ends[2]);

        const int nx {ends[0] - starts[0] + 1 + 2};
        const int ny {ends[1] - starts[1] + 1 + 2};
        const int nz {ends[2] - starts[2] + 1 + 2};

        /* Setup vector types of halo */ 
        int array_of_sizes[]    = {nx, ny, nz};
        int array_of_starts[]   = {0, 0, 0};

        // Front & Back
        MPI_Type_vector(ny-2, nz-2, nz, get_mpi_type<T>(), &vecs[0]);
        MPI_Type_commit(&vecs[0]);

        // Left & Right
        int array_of_subsizes[] = {nx-2, ny-2, 1};
        MPI_Type_create_subarray(dimension, array_of_sizes, array_of_subsizes, array_of_starts,
                                    MPI_ORDER_C, MPI_DOUBLE, &vecs[2]);
        MPI_Type_commit(&vecs[2]);

        // Up & Down
        array_of_subsizes[1] = 1;
        array_of_subsizes[2] = nz-2;
        MPI_Type_create_subarray(dimension, array_of_sizes, array_of_subsizes, array_of_starts,
                                    MPI_ORDER_C, MPI_DOUBLE, &vecs[1]);
        MPI_Type_commit(&vecs[1]);  

        this->resize(nx, ny, nz);
      }

    // exchanges, communications
    public:
      void I_exchange3d();
      void SR_exchange3d();

      void Gather3d(array3d<T>& gather, const int root, MPI_Comm comm);
      
  }; // class array3d_distribute : public array3d<T>


  /**
   * @brief Get the difference between two distributed 1d arrays.
   * 
   * @tparam T The type of elements stored in the arrays.
   * @param ping The first array.
   * @param pong The second array.
   * @return double The difference.
   */
  template <class T>
  double get_difference(const array1d_distribute<T>& ping, const array1d_distribute<T>& pong)
  {
    FINAL_PROJECT_ASSERT_MSG((ping.N == pong.N), "Different Shape!");

    double temp, diff {0.0};
    for (std::size_t i = 0; i < ping.size(); i++)
    {
      temp = ping(i) - pong(i);
      diff += temp * temp;
    }

    return diff;
  }
  
  /**
   * @brief Get the difference between two distributed arrays.
   * 
   * @tparam T The type of elements stored in the arrays.
   * @param ping The first array.
   * @param pong The second array.
   * @return double The difference.
   */
  template <class T>
  double get_difference(const array2d_distribute<T>& ping, const array2d_distribute<T>& pong)
  {
    FINAL_PROJECT_ASSERT_MSG((ping.Rows == pong.Rows && ping.Cols == pong.Cols), "Different Shape!");

    double temp, diff {0.0};
    for (std::size_t i = 0; i < ping.size(); i++)
    {
      temp = ping(i) - pong(i);
      diff += temp * temp;
    }

    return diff;
  }  


  /**
   * @brief Get the difference between two distributed 3d arrays.
   * 
   * @tparam T The type of elements stored in the arrays.
   * @param ping The first array.
   * @param pong The second array.
   * @return double The difference.
   */
  template <class T>
  double get_difference(const array3d_distribute<T>& ping, const array3d_distribute<T>& pong)
  {
    FINAL_PROJECT_ASSERT_MSG((ping.Rows == pong.Rows && ping.Cols == pong.Cols && ping.Height == pong.Height), "Different Shape!");

    double temp, diff {0.0};
    for (std::size_t i = 0; i < ping.size(); i++)
    {
      temp = ping(i) - pong(i);
      diff += temp * temp;
    }

    return diff;
  }

} // namespace final_project






#endif // end of FINAL_PROJECT_ARRAY_DISTRIBUTE_HPP_LIYIHAI