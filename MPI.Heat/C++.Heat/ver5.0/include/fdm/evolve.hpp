#ifndef EVOLVE_HPP
#define EVOLVE_HPP
#pragma once
#include <cmath>
#include <omp.h>

#include "array.hpp"
#include "fdm/heat.hpp"



template <typename T, std::size_t NumDim>
T update_ping_pong1(  final_project::array::array_distribute<T, NumDim> & in, 
                      final_project::array::array_distribute<T, NumDim> & out,
                      final_project::heat_equation<T, NumDim> & hq)
{
  T diff {0};

  // ------------------------------------------------ Update ------------------------------------------------ //
  if (NumDim == 2)
  {
    for (std::size_t i = 1; i < in.get_array().__local_array.__shape[0] - 1; ++i)
    {
      for (std::size_t j = 1; j < in.get_array().__local_array.__shape[1] - 1; ++j)
      {
        T current {in.get_array().__local_array(i,j)};
        out.get_array().__local_array(i,j) = 
          hq.weights[0] * (in.get_array().__local_array(i-1,j) + in.get_array().__local_array(i+1,j))
        + hq.weights[1] * (in.get_array().__local_array(i,j-1) + in.get_array().__local_array(i,j+1))
        + current * (hq.diags[0]*hq.weights[0] + hq.diags[1]*hq.weights[1]);

        diff += std::pow(out.get_array().__local_array(i,j) - in.get_array().__local_array(i,j), 2);
      }
    }
  }

if (NumDim == 3)
{
  for (std::size_t i = 1; i < in.get_array().__local_array.__shape[0] - 1; ++i)
  {
    for (std::size_t j = 1; j < in.get_array().__local_array.__shape[1] - 1; ++j)
    {
      for (std::size_t k = 1; k < in.get_array().__local_array.__shape[2] - 1; ++k)
      {
        T current {in.get_array().__local_array(i,j,k)};
        
        out.get_array().__local_array(i,j,k) = 
            hq.weights[0] * (in.get_array().__local_array(i-1,j,k) + in.get_array().__local_array(i+1,j,k))
          + hq.weights[1] * (in.get_array().__local_array(i,j-1,k) + in.get_array().__local_array(i,j+1,k))
          + hq.weights[2] * (in.get_array().__local_array(i,j,k-1) + in.get_array().__local_array(i,j,k+1))
          + current * (hq.diags[0]*hq.weights[0] + hq.diags[1]*hq.weights[1] + hq.diags[2]*hq.weights[2]);

        diff += std::pow(out.get_array().__local_array(i,j,k) - in.get_array().__local_array(i,j,k), 2);
      }
    }
  }
}
  return diff;
}


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
    std::size_t dim {0};
    {
      auto flag {dim};
      auto n_size {out.get_array().__local_array.__shape[dim]};
      MPI_Sendrecv( &out.get_array().__local_array(1,1), 1, 
                    out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim], flag,
                    &out.get_array().__local_array(n_size-1, 1), 1, 
                    out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag,
                    out.get_topology().__comm_cart, MPI_STATUS_IGNORE);

      MPI_Sendrecv( &out.get_array().__local_array(n_size-2,1), 1, 
                    out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag,
                    &out.get_array().__local_array(0, 1), 1, 
                    out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim], flag,
                    out.get_topology().__comm_cart, MPI_STATUS_IGNORE);

      dim = 1;
      flag = dim;
      n_size = out.get_array().__local_array.__shape[dim];
      MPI_Sendrecv( &out.get_array().__local_array(1,        1), 1, 
                    out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim  ], flag,
                    &out.get_array().__local_array(1, n_size-1), 1, 
                    out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag,
                    out.get_topology().__comm_cart, MPI_STATUS_IGNORE);

      MPI_Sendrecv( &out.get_array().__local_array(1, n_size-2), 1, 
                    out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag,
                    &out.get_array().__local_array(1, 0), 1, 
                    out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim], flag,
                    out.get_topology().__comm_cart, MPI_STATUS_IGNORE);
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

template <typename T, std::size_t NumDim>
T update_omp1(final_project::array::array_distribute<T, NumDim> & in)
{
  // ------------------------------------------------ Update ------------------------------------------------ //
  
  T diff {0}, temp;
  std::size_t off;

  if (NumDim == 2)
  {
    #pragma omp parallel num_threads(2)
    {
    #pragma for private(temp) reduction(+:diff)
    for (std::size_t id = 0; id < 2; ++id)
    {
      for (std::size_t i = 1; i < in.get_array().__local_array.__shape[0] - 1; ++i)
      {
        off = 1 + ( i + id + 1 ) % 2;
        for (std::size_t j = off; j < in.get_array().__local_array.__shape[1] - 1; j+=2)
        {
          temp = in.get_array().__local_array(i,j);
          in.get_array().__local_array(i,j) = 0.25 * (
            in.get_array().__local_array(i+1,j) + in.get_array().__local_array(i-1,j) +
            in.get_array().__local_array(i,j+1) + in.get_array().__local_array(i,j-1)
          );

          diff += std::pow(temp - in.get_array().__local_array(i,j), 2);
        }
      }

      // ------------------------------------------------ Exchange ----------------------------------------------- //
      MPI_Datatype omp_halos[NumDim];
      int array_size[NumDim], array_sub_size[NumDim], array_starts[NumDim];
      for (std::size_t dim = 0; dim < NumDim; ++dim) 
      {
        auto n {in.get_topology().__local_shape[dim] - 2};
        array_starts[dim]   = 0;
        array_size[dim]     = (id == 0) ? (n / 2 + 1) : (n - n / 2);
        array_sub_size[dim] = array_size[dim];
      }

      for (std::size_t dim = 0; dim < NumDim; ++dim)
      {
        auto temp_sub   = array_sub_size[dim];
        array_sub_size[dim] = 1;

        auto temp_size = array_size[dim];
        array_size[dim] = in.get_topology().__local_shape[dim] * 2;

        MPI_Type_create_subarray( in.get_topology().__dimension, 
                                  array_size, array_sub_size, array_starts,
                                  MPI_ORDER_C, in.get_topology().__mpi_value_type, &omp_halos[dim]);

        MPI_Type_commit(&omp_halos[dim]);

        array_sub_size[dim] = temp_sub;
        array_size[dim] = temp_size;
      }

      #pragma omp master
      {
        std::size_t dim {0};
        auto flag {dim};
        auto n_size {in.get_array().__local_array.__shape[dim]};
        MPI_Sendrecv( &in.get_array().__local_array(1,1), 1, 
                      in.get_topology().__halo_vectors[dim], in.get_topology().__neighbors[2*dim], flag,
                      &in.get_array().__local_array(n_size-1, 1), 1, 
                      in.get_topology().__halo_vectors[dim], in.get_topology().__neighbors[2*dim+1], flag,
                      in.get_topology().__comm_cart, MPI_STATUS_IGNORE); // 数据类型错了

        MPI_Sendrecv( &in.get_array().__local_array(n_size-2,1), 1, 
                      in.get_topology().__halo_vectors[dim], in.get_topology().__neighbors[2*dim+1], flag,
                      &in.get_array().__local_array(0, 1), 1, 
                      in.get_topology().__halo_vectors[dim], in.get_topology().__neighbors[2*dim], flag,
                      in.get_topology().__comm_cart, MPI_STATUS_IGNORE);

        dim = 1;
        flag = dim;
        n_size = in.get_array().__local_array.__shape[dim];
        MPI_Sendrecv( &in.get_array().__local_array(1,        1), 1, 
                      in.get_topology().__halo_vectors[dim], in.get_topology().__neighbors[2*dim  ], flag,
                      &in.get_array().__local_array(1, n_size-1), 1, 
                      in.get_topology().__halo_vectors[dim], in.get_topology().__neighbors[2*dim+1], flag,
                      in.get_topology().__comm_cart, MPI_STATUS_IGNORE);

        MPI_Sendrecv( &in.get_array().__local_array(1, n_size-2), 1, 
                      in.get_topology().__halo_vectors[dim], in.get_topology().__neighbors[2*dim+1], flag,
                      &in.get_array().__local_array(1, 0), 1, 
                      in.get_topology().__halo_vectors[dim], in.get_topology().__neighbors[2*dim], flag,
                      in.get_topology().__comm_cart, MPI_STATUS_IGNORE);
      }

      for (std::size_t dim = 0; dim < NumDim; ++dim)
        MPI_Type_free(&omp_halos[dim]);
    }
    }
    diff = std::sqrt(diff / (T)((in.get_array().__local_array.__shape[0]-1) * (in.get_array().__local_array.__shape[1]-1)));
  }

  if (NumDim == 3)
  {      
      #pragma omp parallel for private(temp) num_threads(2) reduction(+:diff)
      for (std::size_t id = 0; id < 2; ++id)
      {
          for (std::size_t i = 1; i < in.get_array().__local_array.__shape[0] - 1; ++i)
          {
              for (std::size_t j = 1; j < in.get_array().__local_array.__shape[1] - 1; ++j)
              {
                  for (std::size_t k = 1; k < in.get_array().__local_array.__shape[2] - 1; ++k)
                  {
                      if ((i + j + k + id) % 2 == 0)
                      {
  temp = in.get_array().__local_array(i,j,k);
  in.get_array().__local_array(i,j,k) = (1.0 / 6.0) * (
      in.get_array().__local_array(i-1,j,k) + in.get_array().__local_array(i+1,j,k) +
      in.get_array().__local_array(i,j-1,k) + in.get_array().__local_array(i,j+1,k) +
      in.get_array().__local_array(i,j,k-1) + in.get_array().__local_array(i,j,k+1));

  diff += std::pow(temp - in.get_array().__local_array(i,j,k), 2);
                      }
                  }
              }
          }
      }
      diff = std::sqrt(diff / (T) (
          (in.get_array().__local_array.__shape[0]-2) * 
          (in.get_array().__local_array.__shape[1]-2) * 
          (in.get_array().__local_array.__shape[2]-2)
      ));
  }

  return diff;
}


template <typename T, std::size_t NumDim>
T update_omp2(final_project::array::array_distribute<T, NumDim> & in, const int & omp_id)
{
  T diff {0};
  // ------------------------------------------------ Update ------------------------------------------------ //
  if (NumDim == 2)
  {
    T temp;
    std::size_t off;
    for (std::size_t i = 1; i < in.get_array().__local_array.__shape[0] - 1; ++i)
    {
      off = 1 + ( i + omp_id + 1 ) % 2;
      for (std::size_t j = off; j < in.get_array().__local_array.__shape[1] - 1; j+=2)
      {
        temp = in.get_array().__local_array(i,j);
        in.get_array().__local_array(i,j) = 0.25 * (
          in.get_array().__local_array(i+1,j) + in.get_array().__local_array(i-1,j) +
          in.get_array().__local_array(i,j+1) + in.get_array().__local_array(i,j-1)
        );

        diff += std::pow(temp - in.get_array().__local_array(i,j), 2);
      }
    }
  }

  return diff;
}




template <typename T, std::size_t NumDim>
void Gather(final_project::array::array_base<T, NumDim>       & gather,   
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