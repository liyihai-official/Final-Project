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
 * @version 3.0
 * @date May 25, 2024
 */

#pragma once
#include <mpi.h>
#include <iostream>

namespace mpi
{
  class env {
    public:
      env() = delete;
      env(int argc, char ** argv){
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);  
      }

      // env(int argc, char ** argv, int required, int * provided){
      //   MPI_Init_thread(&argc, &argv, required, provided);
      //   MPI_Comm_size(MPI_COMM_WORLD, &size_);
      //   MPI_Comm_rank(MPI_COMM_WORLD, &rank_);  
      // }

      env(const env &)  = delete;
      env(env &&)       = delete;

      env& operator=(const env&) = delete;
      env& operator=(env&&)      = delete;

      ~env() {
        MPI_Finalize();
      }

      int rank() {return rank_;}
      int size() {return size_;}

    private:
      int rank_, size_;
  };
}; // namespace mpi