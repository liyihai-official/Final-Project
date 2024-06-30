#include <iostream>
#include "mpi_detials/mpi_topology.hpp"
#include "mpi_detials/mpi_types.hpp"
#include "mpi_detials/mpi_environment.hpp"
#include "multi_array/base.hpp"
#include "mpi_distribute/mpi_distribute_array.hpp"
#include "array.hpp"

#include <cmath>

#include <omp.h>
#include <vector>
#include <cstring>
// template <typename T, std::size_t NumDims>
// struct heat_equation {

//   heat_equation(final_project::__detail::__types::__multi_array_shape<NumDims> global_grid_shape)
//   {
//     for (std::size_t i = 0; i < NumDims; ++i)
//     {
//       minRange[i] = 0;
//       maxRange[i] = 1;
//       deltaXs[i] = (maxRange[i] - minRange[i]) / global_grid_shape[i];

//       dt = 1.0 / std::pow(2, NumDims);
//       dt = std::min(dt, 0.1);

//       weights[i] = coff * dt / (deltaXs[i] * deltaXs[i]);
//       diags[i] = -2.0 + (deltaXs[i] * deltaXs[i]) / (NumDims * coff * dt);
//     }
//   }

//   private:
//   T coff {1};
//   T dt {0.1}; 
//   T diags[NumDims], weights[NumDims], minRange[NumDims], maxRange[NumDims], deltaXs[NumDims];

// };


template <typename T, std::size_t NumDim>
void exchange( final_project::array::array_distribute<T, NumDim> & out )
{
  if (NumDim == 2)
  {
    std::size_t dim {0};
    // for (std::size_t dim = 0; dim < 1; ++dim)
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
}




template <typename T, std::size_t NumDim>
T update(final_project::array::array_distribute<T, NumDim> & in)
{
  T diff {0}, temp;
  std::size_t off;

  if (NumDim == 2)
  {
    #pragma omp parallel for private(temp) num_threads(2) reduction(+:diff)
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
    }
    diff = std::sqrt(diff / (T)((in.get_array().__local_array.__shape[0]-1) * (in.get_array().__local_array.__shape[1]-1)));
  }
  return diff;
}


template <typename T, std::size_t NumDim>
T update( final_project::array::array_distribute<T, NumDim> & in, 
          final_project::array::array_distribute<T, NumDim> & out)
{
  T diff {0};

  if (NumDim == 2)
  {
    for (std::size_t i = 1; i < in.get_array().__local_array.__shape[0] - 1; ++i)
    {
      for (std::size_t j = 1; j < in.get_array().__local_array.__shape[1] - 1; ++j)
      {
        out.get_array().__local_array(i,j) = 0.25 * (
          in.get_array().__local_array(i+1,j) + in.get_array().__local_array(i-1,j) +
          in.get_array().__local_array(i,j+1) + in.get_array().__local_array(i,j-1)
        );

        diff += std::pow(out.get_array().__local_array(i,j) - in.get_array().__local_array(i,j), 2);
      }
    }

  }

  diff = std::sqrt(diff / (T)((in.get_array().__local_array.__shape[0]-1) * (in.get_array().__local_array.__shape[1]-1)));

  if (NumDim == 2)
  {
    std::size_t dim {0};
    // for (std::size_t dim = 0; dim < 1; ++dim)
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

int main( int argc, char ** argv)
{
  auto world {final_project::mpi::env(argc, argv)};


  auto shape {final_project::__detail::__types::__multi_array_shape<2>(19, 17)};
  // auto an_topology {final_project::__detail::__mpi_types::__mpi_topology<double, 2>(shape, world)};
  // std::cout 
  // << " PROCESS " << an_topology.__rank 
  // << " Has Coordinate : \t ["
  // << an_topology.__coordinates[0] << ", "
  // << an_topology.__coordinates[1] << ", "
  // << an_topology.__coordinates[2] << "]"
  // << " \t "
  // << " Has Shape : \t [" 
  // << an_topology.__local_shape[0] << ", "
  // << an_topology.__local_shape[1] << ", "
  // << an_topology.__local_shape[2] << "] "
  // << " Range : \t [ ("
  // << an_topology.__starts[0] << ", " << an_topology.__ends[0] << ") " << ", ("
  // << an_topology.__starts[1] << ", " << an_topology.__ends[1] << ") " << ", ("
  // << an_topology.__starts[2] << ", " << an_topology.__ends[2] << ") " << "] "
  // << std::endl;


  // auto Array {final_project::__detail::__multi_array::__array<float, 2>(shape)};
  // Array.fill(0);

  // if (world.rank() == 0 )
  //   std::cout << Array << std::endl;
  // }


  // auto DA {final_project::__detail::__mpi_distribute_array<double, 2>(shape, world)};
  // DA.__fill_boundary(1);

  // if (world.rank() == 0) 
  // {
  // std::cout << DA << std::endl;
  // }

  // heat_equation<double, 2> equation {shape};
  if (world.rank() == 0)
  { }

  // 
  auto DD {final_project::array::array_distribute<double, 2>(shape, world)};
  auto GG {final_project::array::array_distribute<double, 2>(shape, world)};
  DD.fill_boundary(1);
  GG.fill_boundary(1);

  // std::cout << DD.get_array() << std::endl;
  double diff {0}, gdiff {0};

  auto t1 = MPI_Wtime();
  for (std::size_t i = 0; i < 4; ++i)
  {
    // diff = update(DD);
    // diff = DD.update();

    diff = update(DD, GG);
    // exchange(GG);
    // MPI_Reduce(&diff, &gdiff, 1, MPI_DOUBLE, MPI_SUM, 0, world.comm());
    // if (world.rank() == 0 && i % 10 == 0) std::cout << std::fixed << std::setprecision(15) << std::setw(15) << gdiff << std::endl;

    diff = update(GG, DD);
    // exchange(DD);
    // MPI_Reduce(&diff, &gdiff, 1, MPI_DOUBLE, MPI_SUM, 0, world.comm());

  }
  auto t2 = MPI_Wtime();
  // update(GG, DD);
  std::cout << DD.get_array() << std::endl;

  MPI_Barrier(world.comm());
  // exchange(DD);
  // MPI_Barrier(world.comm());
  // std::cout << DD.get_array() << std::endl;
  // auto G {final_project::array::array_base<double, 2>(shape)};

  // Gather(G, DD);
  // if (world.rank() == 0) std::cout << G.get_array() << std::endl;
  // if (world.rank() == 0) G.saveToBinaryFile("TEST.bin");

  // std::cout << t2 - t1 << std::endl;
  
  // std::cout 
  // << "PROC : " << DD.get_topology().__rank 
  // << " (" << DD.get_topology().__rank << ",1) >>> " 
  // << DD.get_array().__local_array(DD.get_topology().__rank,1) 
  // << std::endl;
  
  

  return 0;
}