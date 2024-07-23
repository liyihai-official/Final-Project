#ifndef EVOLVE_HPP
#define EVOLVE_HPP

#pragma once
#include <cmath>
#include <omp.h>

#include "array.hpp"
#include "fdm/heat.hpp"

#include "evolve/evolve_pure_mpi.hpp"

#include "evolve/evolve_hybrid_1.hpp"
#include "evolve/evolve_hybrid_2.hpp"


/// @brief 
/// @tparam T 
/// @tparam NumDim 
/// @param out 
template <typename T, std::size_t NumDim>
void exchange_ping_pong1(final_project::array::array_distribute<T, NumDim> & out)
{
  // ------------------------------------------------ Exchange ----------------------------------------------- //
  if (NumDim == 2)
  {
    // std::size_t dim {0};
    // {
    //   auto flag {dim};
    //   auto n_size {out.get_array().__local_array.__shape[dim]};
    //   MPI_Sendrecv( &out.get_array().__local_array(1,1), 1, 
    //                 out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim], flag,
    //                 &out.get_array().__local_array(n_size-1, 1), 1, 
    //                 out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag,
    //                 out.get_topology().__comm_cart, MPI_STATUS_IGNORE);

    //   MPI_Sendrecv( &out.get_array().__local_array(n_size-2,1), 1, 
    //                 out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag,
    //                 &out.get_array().__local_array(0, 1), 1, 
    //                 out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim], flag,
    //                 out.get_topology().__comm_cart, MPI_STATUS_IGNORE);

    //   dim = 1;
    //   flag = dim;
    //   n_size = out.get_array().__local_array.__shape[dim];
    //   MPI_Sendrecv( &out.get_array().__local_array(1,        1), 1, 
    //                 out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim  ], flag,
    //                 &out.get_array().__local_array(1, n_size-1), 1, 
    //                 out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag,
    //                 out.get_topology().__comm_cart, MPI_STATUS_IGNORE);

    //   MPI_Sendrecv( &out.get_array().__local_array(1, n_size-2), 1, 
    //                 out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag,
    //                 &out.get_array().__local_array(1, 0), 1, 
    //                 out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim], flag,
    //                 out.get_topology().__comm_cart, MPI_STATUS_IGNORE);
    // }

std::size_t dim {0};
{
    MPI_Request requests[8];
    int request_count = 0;

    auto flag {dim};
    auto n_size {out.get_array().__local_array.__shape[dim]};
    
    // Send to neighbor 2*dim and receive from neighbor 2*dim+1
    MPI_Isend(&out.get_array().__local_array(1,         1), 1, 
              out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim], flag, 
              out.get_topology().__comm_cart, &requests[request_count++]);

    MPI_Irecv(&out.get_array().__local_array(n_size-1,  1), 1, 
              out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag, 
              out.get_topology().__comm_cart, &requests[request_count++]);

    // Send to neighbor 2*dim+1 and receive from neighbor 2*dim
    MPI_Isend(&out.get_array().__local_array(n_size-2,  1), 1, 
              out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag, 
              out.get_topology().__comm_cart, &requests[request_count++]);

    MPI_Irecv(&out.get_array().__local_array(0,         1), 1, 
              out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim], flag, 
              out.get_topology().__comm_cart, &requests[request_count++]);

    dim = 1;
    flag = dim;
    n_size = out.get_array().__local_array.__shape[dim];

    // Send to neighbor 2*dim and receive from neighbor 2*dim+1
    MPI_Isend(&out.get_array().__local_array(1,         1), 1, 
              out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim], flag, 
              out.get_topology().__comm_cart, &requests[request_count++]);

    MPI_Irecv(&out.get_array().__local_array(1,   n_size-1), 1, 
              out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag, 
              out.get_topology().__comm_cart, &requests[request_count++]);

    // Send to neighbor 2*dim+1 and receive from neighbor 2*dim
    MPI_Isend(&out.get_array().__local_array(1,   n_size-2), 1, 
              out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag, 
              out.get_topology().__comm_cart, &requests[request_count++]);

    MPI_Irecv(&out.get_array().__local_array(1,         0), 1, 
              out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim], flag, 
              out.get_topology().__comm_cart, &requests[request_count++]);

    // Wait for all non-blocking operations to complete
    MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
}
  }

  
  if (NumDim == 3)
  {
    std::size_t dim {0};
    {
      auto flag {dim};
      auto n_size {out.get_array().__local_array.__shape[dim]};

      MPI_Sendrecv( &out.get_array().__local_array(1,1,1), 1,
                    out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim], flag,
                    &out.get_array().__local_array(n_size-1, 1, 1), 1,
                    out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag,
                    out.get_topology().__comm_cart, MPI_STATUS_IGNORE);

      MPI_Sendrecv( &out.get_array().__local_array(n_size-2,1,1), 1,
                    out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag,
                    &out.get_array().__local_array(0,1,1), 1,
                    out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim], flag,
                    out.get_topology().__comm_cart, MPI_STATUS_IGNORE);

      dim = 1;
      flag = dim;
      n_size = out.get_array().__local_array.__shape[dim];
      MPI_Sendrecv( &out.get_array().__local_array(1,        1, 1), 1, 
                          out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim  ], flag,
                    &out.get_array().__local_array(1, n_size-1,1), 1, 
                          out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag,
                    out.get_topology().__comm_cart, MPI_STATUS_IGNORE);

      MPI_Sendrecv( &out.get_array().__local_array(1, n_size-2,1), 1, 
                    out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag,
                    &out.get_array().__local_array(1, 0, 1), 1, 
                    out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim], flag,
                    out.get_topology().__comm_cart, MPI_STATUS_IGNORE);

      dim = 2;
      flag = dim;
      n_size = out.get_array().__local_array.__shape[dim];
      MPI_Sendrecv( &out.get_array().__local_array(1,        1, 1), 1, 
                          out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim  ], flag,
                    &out.get_array().__local_array(1, 1, n_size-1), 1, 
                          out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag,
                    out.get_topology().__comm_cart, MPI_STATUS_IGNORE);

      MPI_Sendrecv( &out.get_array().__local_array(1, 1, n_size-2), 1, 
                          out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag,
                    &out.get_array().__local_array(1, 1, 0), 1, 
                          out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim], flag,
                    out.get_topology().__comm_cart, MPI_STATUS_IGNORE);
    }
  }
}



#endif // end define EVOLVE_HPP