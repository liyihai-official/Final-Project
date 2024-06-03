/**
 * @file gather.cpp
 * @brief This file contains the implementation of the Gather2d function 
 *        for gathering distributed 2D arrays in an MPI environment.
 * 
 * This header file defines the function Gather2d for the array2d_distribute 
 * class, which is used to gather distributed 2D arrays from all MPI 
 * processes into a single array on the root process.
 * 
 * @author LI Yihai
 * @version 3.0
 * @date May 25, 2024
 */

#ifndef FINAL_PROJECT_GATHER_HPP_LIYIHAI
#define FINAL_PROJECT_GATHER_HPP_LIYIHAI
#include "multi_array/array_distribute.cpp"
#include <cstring>
#include <limits>

namespace final_project
{

  /**
   * @brief Gather the distributed 1D arrays from all processes to a single array on 
   *        the root process.
   * 
   * This function gathers the distributed parts of a 1D array from all MPI processes
   * and combines them into a single 1D array on the root process.
   * 
   * @tparam T The type of the elements in the array.
   * @param gather The array to gather the data into (only relevant on the root process).
   * @param root The rank of the root process.
   * @param comm The MPI communicator.
   */
  template <class T>
  void array1d_distribute<T>::Gather1d(array1d<T>& gather, const int root, MPI_Comm comm)
  {
    MPI_Datatype mpi_T {get_mpi_type<T>()};

    std::size_t N {this->size() - 2};

    int pid, i;
    int s_list[num_proc], N_list[num_proc];

    // Add Boundary Conditions
    if (rank == 0) --starts[0], ++N;
    if (rank == num_proc -1) ++N;

    MPI_Gather(starts, 1, MPI_INT, s_list, 1, MPI_INT, root, comm);

    MPI_Gather(&N,     1, MPI_INT, N_list, 1, MPI_INT, root, comm);

    if (rank != root)
    {
      MPI_Send(&(*this)(1), N, mpi_T, root, rank, comm);
    }

    if (rank == root)
    {
      for (pid = 0; pid < num_proc; ++pid)
      {
        if (pid == root)
        {
          memcpy(&gather(starts[0]), &(*this)(starts[0]), N_list[pid]*sizeof(T));
        }

        if (pid != root)
        {
          MPI_Recv(&gather(s_list[pid]), N_list[pid], mpi_T, pid, pid, comm, MPI_STATUS_IGNORE);
        }
      }
    }
  } // array1d_distribute<T>::Gather1d

  /**
   * @brief Gather the distributed 2D arrays from all processes to a single array on 
   *        the root process.
   * 
   * This function gathers the distributed parts of a 2D array from all MPI processes
   * and combines them into a single 2D array on the root process.
   * 
   * @tparam T The type of the elements in the array.
   * @param gather The array to gather the data into (only relevant on the root process).
   * @param root The rank of the root process.
   * @param comm The MPI communicator.
   */
  template <class T>
  void array2d_distribute<T>::Gather2d(array2d<T>& gather, const int root, MPI_Comm comm)
  {
    MPI_Datatype temp, Block, mpi_T {get_mpi_type<T>()};

    std::size_t Nx {this->rows() - 2}, Ny {this->cols() - 2};

    std::size_t i_idx {1}, j_idx{1};

    int pid, i;
    int s0_list[num_proc], s1_list[num_proc], nx_list[num_proc], ny_list[num_proc];

    // Add Boundaries
    // Up
    if (starts[0] == 1)
    {
      -- starts[0];
      -- i_idx;
      ++ Nx;
    }

    // Left
    if (starts[1] == 1)
    {
      -- starts[1];
      -- j_idx;
      ++ Ny;
    }

    if (ends[0] == glob_Rows - 2) ++ Nx; // Down
    if (ends[1] == glob_Cols - 2) ++ Ny; // Right


    MPI_Type_vector(Nx, Ny, this->cols(), mpi_T, &Block);
    MPI_Type_commit(&Block);

    MPI_Gather(&starts[0], 1, MPI_INT, s0_list, 1, MPI_INT, root, comm);
    MPI_Gather(&starts[1], 1, MPI_INT, s1_list, 1, MPI_INT, root, comm);

    // narrowing : std::size_t --->>  MPI_INT
    MPI_Gather(&Nx       , 1, MPI_INT, nx_list, 1, MPI_INT, root, comm);
    MPI_Gather(&Ny       , 1, MPI_INT, ny_list, 1, MPI_INT, root, comm);

    if (rank != root)
    {
      MPI_Send(&(*this)(i_idx,j_idx), 1, Block, root, rank, comm);
    }
    MPI_Type_free(&Block);

    if (rank == root)
    {
      for (pid = 0; pid < num_proc; ++pid)
      {
        if (pid == root)
        {
          for (i = starts[0]; i <= ends[0]; ++i)
          {
            memcpy(&gather(i, starts[1]), &(*this)(i, starts[1]), ny_list[pid]*sizeof(T));
          }
        }

        if (pid != root)
        {
          MPI_Type_vector(nx_list[pid], ny_list[pid], gather.cols(), mpi_T, &temp);
          MPI_Type_commit(&temp);

          MPI_Recv( &gather(s0_list[pid], s1_list[pid]), 1,
                    temp, pid, pid, comm, MPI_STATUS_IGNORE);

          MPI_Type_free(&temp);  
        }
      }
    }

  } // array2d_distribute<T>::Gather2d


  /**
   * @brief Gather the distributed 3D arrays from all processes to a single array on 
   *        the root process.
   * 
   * This function gathers the distributed parts of a 3D array from all MPI processes
   * and combines them into a single 3D array on the root process.
   * 
   * @tparam T The type of the elements in the array.
   * @param gather The array to gather the data into (only relevant on the root process).
   * @param root The rank of the root process.
   * @param comm The MPI communicator.
   */
  template <class T>
  void array3d_distribute<T>::Gather3d(array3d<T>& gather, const int root, MPI_Comm comm)
  {
    MPI_Datatype Block, mpi_T {get_mpi_type<T>()};
    
    const int Nx {ends[0] - starts[0] + 1};
    const int Ny {ends[1] - starts[1] + 1};
    const int Nz {ends[2] - starts[2] + 1};

    int pid, i, j, k;
    int s0_list[num_proc], s1_list[num_proc], s2_list[num_proc];
    int nx_list[num_proc], ny_list[num_proc], nz_list[num_proc];

    MPI_Gather(&starts[0], 1, MPI_INT, s0_list, 1, MPI_INT, root, comm);
    MPI_Gather(&starts[1], 1, MPI_INT, s1_list, 1, MPI_INT, root, comm);
    MPI_Gather(&starts[2], 1, MPI_INT, s2_list, 1, MPI_INT, root, comm);

    // narrowing : std::size_t --->>  MPI_INT
    MPI_Gather(&Nx       , 1, MPI_INT, nx_list, 1, MPI_INT, root, comm);
    MPI_Gather(&Ny       , 1, MPI_INT, ny_list, 1, MPI_INT, root, comm);
    MPI_Gather(&Nz       , 1, MPI_INT, nz_list, 1, MPI_INT, root, comm);

    if (rank != root)
    {
      int array_of_sizes[]    = {Nx+2, Ny+2, Nz+2};
      int array_of_subsizes[] = {Nx, Ny, Nz};
      int array_of_starts[]   = {0, 0, 0};
      MPI_Type_create_subarray(dimension, array_of_sizes, array_of_subsizes, array_of_starts,
                                  MPI_ORDER_C, MPI_DOUBLE, &Block);
      MPI_Type_commit(&Block);

      MPI_Send(&(*this)(1,1,1), 1, Block, root, rank, comm);
    }

    if (rank == root)
    {
      for (pid = 0; pid < num_proc; ++pid)
      {
        if (pid == root)
        {
          for ( i = starts[0]; i <= ends[0]; ++i)
          {
            for ( j = starts[1]; j <= ends[1]; ++j)
            {
              memcpy( &gather( i, j, starts[2]), 
                      &(*this)(i, j, starts[2]), nz_list[pid]*sizeof(T));
            }
          }
        }

        if (pid != root)
        {
          if (
            gather.Rows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            gather.Cols > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            gather.Height > static_cast<std::size_t>(std::numeric_limits<int>::max())) 
          {
            throw std::overflow_error("Size exceeds the range of int");
          }

          int array_of_sizes[] = {
            static_cast<int>(gather.Rows),
            static_cast<int>(gather.Cols),
            static_cast<int>(gather.Height)
          };
          int array_of_subsizes[] = {nx_list[pid]  , ny_list[pid]  , nz_list[pid]  };
          int array_of_starts[]   = {0, 0, 0};
          MPI_Type_create_subarray(dimension, array_of_sizes, array_of_subsizes, array_of_starts,
                                      MPI_ORDER_C, MPI_DOUBLE, &Block);
          MPI_Type_commit(&Block);

          MPI_Recv(&gather( s0_list[pid], s1_list[pid], s2_list[pid]), 1, Block, pid, 
                                          pid, comm, MPI_STATUS_IGNORE);
          MPI_Type_free(&Block);
        }
      }
    }
  } // array3d_distribute<T>::Gather3d

} // namespace final_project


#endif