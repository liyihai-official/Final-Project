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
#include "array.cpp"
#include <cstring>

namespace final_project
{

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

    int pid, i;
    int s0_list[num_proc], s1_list[num_proc], nx_list[num_proc], ny_list[num_proc];

    MPI_Type_vector(Nx, Ny, Ny+2, mpi_T, &Block);
    MPI_Type_commit(&Block);

    MPI_Gather(&starts[0], 1, MPI_INT, s0_list, 1, MPI_INT, root, comm);
    MPI_Gather(&starts[1], 1, MPI_INT, s1_list, 1, MPI_INT, root, comm);
    MPI_Gather(&Nx       , 1, MPI_INT, nx_list, 1, MPI_INT, root, comm);
    MPI_Gather(&Ny       , 1, MPI_INT, ny_list, 1, MPI_INT, root, comm);

    if (rank != root)
    {
      MPI_Send(&(*this)(1,1), 1, Block, root, rank, comm);
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

          MPI_Recv(&gather(s0_list[pid], s1_list[pid]), 1,
                            temp, pid, pid, comm, MPI_STATUS_IGNORE);

          MPI_Type_free(&temp);  
        }
      }
    }

    
  }
  

} // namespace final_project


#endif