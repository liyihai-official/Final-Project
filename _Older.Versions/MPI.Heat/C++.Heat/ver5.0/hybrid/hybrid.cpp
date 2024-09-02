#include <iostream>

#include <cmath>

#include <vector>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "fdm/evolve.hpp"

template <typename T, std::size_t NumDim>
void update_ping_pong3(  final_project::array::array_distribute<T, NumDim> & in, 
                        final_project::array::array_distribute<T, NumDim> & out)
{
  if (NumDim == 2)
  {
    #pragma omp for
    for (std::size_t i = 1; i < in.get_array().__local_array.__shape[0] - 1; ++i)
    {
      for (std::size_t j = 1; j < in.get_array().__local_array.__shape[1] - 1; ++j)
      {
        out.get_array().__local_array(i,j) = 0.25 * (
      in.get_array().__local_array(i+1,j) + in.get_array().__local_array(i-1,j) +
      in.get_array().__local_array(i,j+1) + in.get_array().__local_array(i,j-1)
        );
      }
    }
  } 
}


template <typename T, std::size_t NumDim>
void get_difference3( final_project::array::array_distribute<T, NumDim> & in,
                      final_project::array::array_distribute<T, NumDim> & out, T & diff)
{
  for (std::size_t i = 1; i < in.get_array().__local_array.__shape[0]-1; ++i)
  {
    for (std::size_t j = 1; j < in.get_array().__local_array.__shape[1]-1; ++j)
    {
      diff += std::pow(out.get_array().__local_array(i,j) - in.get_array().__local_array(i,j), 2);
    }
  }
}




template <typename T, std::size_t NumDim>
void exchange(  final_project::array::array_distribute<T, NumDim> & out )
{
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
    MPI_Isend(&out.get_array().__local_array(1,1), 1, 
              out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim], flag, 
              out.get_topology().__comm_cart, &requests[request_count++]);

    MPI_Irecv(&out.get_array().__local_array(n_size-1, 1), 1, 
              out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag, 
              out.get_topology().__comm_cart, &requests[request_count++]);

    // Send to neighbor 2*dim+1 and receive from neighbor 2*dim
    MPI_Isend(&out.get_array().__local_array(n_size-2,1), 1, 
              out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag, 
              out.get_topology().__comm_cart, &requests[request_count++]);

    MPI_Irecv(&out.get_array().__local_array(0, 1), 1, 
              out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim], flag, 
              out.get_topology().__comm_cart, &requests[request_count++]);

    dim = 1;
    flag = dim;
    n_size = out.get_array().__local_array.__shape[dim];

    // Send to neighbor 2*dim and receive from neighbor 2*dim+1
    MPI_Isend(&out.get_array().__local_array(1, 1), 1, 
              out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim], flag, 
              out.get_topology().__comm_cart, &requests[request_count++]);

    MPI_Irecv(&out.get_array().__local_array(1, n_size-1), 1, 
              out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag, 
              out.get_topology().__comm_cart, &requests[request_count++]);

    // Send to neighbor 2*dim+1 and receive from neighbor 2*dim
    MPI_Isend(&out.get_array().__local_array(1, n_size-2), 1, 
              out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag, 
              out.get_topology().__comm_cart, &requests[request_count++]);

    MPI_Irecv(&out.get_array().__local_array(1, 0), 1, 
              out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim], flag, 
              out.get_topology().__comm_cart, &requests[request_count++]);

    // Wait for all non-blocking operations to complete
    MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
}

  }
}




int main( int argc, char ** argv)
{
  const int nsteps {200000};

  auto world {final_project::mpi::env(argc, argv)};
  auto shape {final_project::__detail::__types::__multi_array_shape<2>(1000, 1000)};

  auto DD {final_project::array::array_distribute<double, 2>(shape, world)};
  auto GG {final_project::array::array_distribute<double, 2>(shape, world)};
  DD.fill_boundary(10);
  GG.fill_boundary(10);

  double ldiff {0}, gdiff {0}, t_com {0};
  auto G {final_project::array::array_base<double,2>(shape)};
  int num_threads = 1;

  #pragma omp parallel num_threads(2)
  {
#ifdef _OPENMP
    #pragma omp master
    num_threads = omp_get_num_threads();
#endif

    #pragma omp single
    {
      if ( 0 == world.rank() )
      {
        std::cout << "Simulation parameters: "
                  << "ROWS: " << shape[0] << " COLS: " << shape[1]
                  << " Time Steps: " << nsteps << std::endl;
        
        std::cout << "Number of MPI Tasks: " << world.size() << std::endl;
        std::cout << "Number of OpenMP Threads: " << num_threads << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Average Temperature at start: " << 0 << std::endl;
      }
    }
 
  auto start_clock {MPI_Wtime()};
  for (int iter = 1; iter < 3760; iter++)
  {
    #pragma omp single
    exchange(DD);
    update_ping_pong3(DD, GG);

    #pragma omp single
    {
      get_difference3(DD,GG, ldiff);
      MPI_Allreduce(&ldiff, &gdiff, 1, MPI_DOUBLE, MPI_SUM, world.comm());
      if (world.rank() == 0 && iter % 100 == 0) 
          std::cout   << "Iteration: " << std::fixed << std::setw(5) << iter << "\t" 
                      << std::fixed << std::setprecision(10) << gdiff << std::endl;
      DD.swap(GG);
    } // omp end single


    // if (gdiff <= 1E-4) break;
    ldiff = 0;    
  }
  auto stop_clock {MPI_Wtime()};



  #pragma omp master
  {
    Gather(G, DD);
    // if (world.rank() == 0) std::cout << G.get_array() << std::endl;
    if (world.rank() == 0) G.saveToBinaryFile("TEST.bin");
    // MPI_Reduce(&t2, &t_com, 1, MPI_DOUBLE, MPI_SUM, 0, world.comm());
    if (world.rank() == 0) std::cout << stop_clock - start_clock << std::endl;
  } // omp end master

  } // omp end parallel

  Gather(G, DD);
  // // if (world.rank() == 0) std::cout << G.get_array() << std::endl;
  if (world.rank() == 0) G.saveToBinaryFile("TEST.bin");
  // // MPI_Reduce(&t2, &t_com, 1, MPI_DOUBLE, MPI_SUM, 0, world.comm());
  // if (world.rank() == 0) std::cout << stop_clock - start_clock << std::endl;

  
  return 0;
}