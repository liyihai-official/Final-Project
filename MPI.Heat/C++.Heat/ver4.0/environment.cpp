/**
 * @file environment.cpp
 * @brief This file contains the definition of the mpi::env class for 
 *        managing MPI environment initialization and finalization.
 * 
 * This header file defines the mpi::env class, which is responsible for 
 * initializing and finalizing the MPI environment, as well as providing 
 * the rank and size of the MPI communicator.
 * 
 * @author LI Yihai
 * @version 4.0
 * @date Jun 7, 2024
 */

#pragma once
#include <mpi.h>
#include <iostream>


namespace final_project {
namespace mpi
{
  class env {
    public:
      env() = delete;
      env(int argc, char ** argv)
        : comm_ {MPI_COMM_WORLD}
      {
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);  
      }

      env(MPI_Comm comm)
       : comm_ {comm}
      {
        MPI_Comm_size(comm_, &size_);
        MPI_Comm_rank(comm_, &rank_);
      }

      env(const env &)  = delete;
      env(env &&)       = delete;

      env& operator=(const env&) = delete;
      env& operator=(env&&)      = delete;

      ~env() {
        MPI_Finalize();
      }

      int rank()      {return rank_;}
      int size()      {return size_;}
      MPI_Comm comm() {return comm_;}

    private:
      int rank_, size_;
      MPI_Comm comm_;
  };
} // namespace mpi
} // namespace final_project