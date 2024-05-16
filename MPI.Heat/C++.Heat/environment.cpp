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