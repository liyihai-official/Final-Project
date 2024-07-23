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




/// @brief 
/// @tparam T 
/// @tparam NumDim 
/// @param gather 
/// @param array 
template <typename T, std::size_t NumDim>
void Gather(
  final_project::array::array_base<T, NumDim>       & gather,   
  final_project::array::array_distribute<T, NumDim> & array)
{
  MPI_Datatype sbuf_block, temp, mpi_T {final_project::__detail::__mpi_types::__get_mpi_type<T>()};

  int root {0};

  int indexs[NumDim];
  
  int num_procs {array.get_topology().__num_procs};
  int pid, i, j, k;

  int Ns[NumDim], starts_cpy[NumDim];

  int s_list[NumDim][num_procs], n_list[NumDim][num_procs];

  int array_of_sizes[NumDim], array_of_starts[NumDim], array_of_subsizes[NumDim];

  for (std::size_t dim = 0; dim < NumDim; ++dim)
  {
    indexs[dim] = 1;
    Ns[dim] = array.get_topology().__ends[dim] - array.get_topology().__starts[dim] + 1; 
    starts_cpy[dim] = array.get_topology().__starts[dim];

    if (starts_cpy[dim] == 1) 
    {
      -- starts_cpy[dim];
      -- indexs[dim];
      ++ Ns[dim];
    }

    if (array.get_topology().__ends[dim] == array.get_topology().__global_shape[dim] - 2) 
      ++ Ns[dim];
    
    MPI_Gather(&starts_cpy[dim], 1, MPI_INT, s_list[dim], 1, MPI_INT, root, array.get_topology().__comm_cart);
    MPI_Gather(&Ns[dim], 1, MPI_INT, n_list[dim], 1, MPI_INT, root, array.get_topology().__comm_cart);
  }

  if (array.get_topology().__rank != root)
  {
    for (std::size_t dim = 0; dim < NumDim; ++dim)
    {
      array_of_sizes[dim] = array.get_topology().__local_shape[dim];
      array_of_subsizes[dim] = Ns[dim];
      array_of_starts[dim] = 0;
    }

    MPI_Type_create_subarray( array.get_topology().__dimension, 
                              array_of_sizes,
                              array_of_subsizes,
                              array_of_starts,
                              MPI_ORDER_C, mpi_T, &sbuf_block);

    MPI_Type_commit(&sbuf_block);

    if (NumDim == 2)
      MPI_Send( &(array.get_array().__local_array(indexs[0], indexs[1])), 
                1, sbuf_block, root, 
                array.get_topology().__rank, 
                array.get_topology().__comm_cart);

    if (NumDim == 3)
      MPI_Send( &(array.get_array().__local_array(indexs[0], indexs[1], indexs[2])), 
                1, sbuf_block, root, 
                array.get_topology().__rank, 
                array.get_topology().__comm_cart);

    MPI_Type_free(&sbuf_block);
  }


  if (array.get_topology().__rank == root)
  {
    for (pid = 0; pid < num_procs; ++pid)
    {
      if (pid == root)
      {
        for ( i=starts_cpy[0]; i <= array.get_topology().__ends[0]; ++i) 
        {
          if (NumDim == 2) 
            memcpy( &gather.get_array()(i, starts_cpy[1]), 
                    &array.get_array().__local_array(i, starts_cpy[1]), n_list[1][pid]*sizeof(T));

          if (NumDim == 3) {
            for ( j=starts_cpy[1]; j <= array.get_topology().__ends[1]; ++j)
              memcpy( &gather.get_array()(i, j, starts_cpy[2]), 
                      &array.get_array().__local_array(i, j, starts_cpy[2]), n_list[2][pid]*sizeof(T));
          }
        }        
      }

      if (pid != root) 
      {
        for (std::size_t dim = 0; dim < NumDim; ++dim)
        {
          array_of_starts[dim] = 0;
          array_of_sizes[dim] = array.get_topology().__global_shape[dim];
          array_of_subsizes[dim] = n_list[dim][pid];
        }

        MPI_Type_create_subarray( array.get_topology().__dimension, 
                                  array_of_sizes, 
                                  array_of_subsizes,
                                  array_of_starts,
                                  MPI_ORDER_C, mpi_T, &temp);

        MPI_Type_commit(&temp);

        // s_list[NumDim][num_procs]
        if (NumDim == 2)
          MPI_Recv( &gather.get_array()(s_list[0][pid], s_list[1][pid]), 
                    1, temp, pid, pid, 
                    array.get_topology().__comm_cart,
                    MPI_STATUS_IGNORE);

        if (NumDim == 3)
          MPI_Recv( &gather.get_array()(s_list[0][pid], s_list[1][pid], s_list[2][pid]), 
                    1, temp, pid, pid, 
                    array.get_topology().__comm_cart,
                    MPI_STATUS_IGNORE);

        MPI_Type_free(&temp);
      }
    }
  }

}


#endif // end define EVOLVE_HPP