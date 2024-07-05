#include <iostream>

#include <cmath>

#include <omp.h>
#include <vector>
#include <cstring>

#include "update.hpp"


int main( int argc, char ** argv)
{
  auto world {final_project::mpi::env(argc, argv)};

  auto shape {final_project::__detail::__types::__multi_array_shape<2>(9, 11)};

  auto DD {final_project::array::array_distribute<double, 2>(shape, world)};
  DD.fill_boundary(10);
  // std::cout << DD.get_array() << std::endl;


  double diff {0}, gdiff {0}, t_com {0};
  auto G {final_project::array::array_base<double,2>(shape)};


  double t1 {MPI_Wtime()};
  #pragma omp parallel num_threads(2)
  {
    const int id {omp_get_thread_num()};
    const std::size_t NumDim {2};
    MPI_Datatype Halos[NumDim];
    int array_size[NumDim], array_sub_size[NumDim], array_starts[NumDim];
    
    for (std::size_t dim = 0; dim < NumDim; ++dim) 
    {
      auto n {DD.get_topology().__local_shape[dim] - 2};
      auto half_n {static_cast<int>(std::round(static_cast<double>(n) / 2.0))};
      array_starts[dim]   = 0;
      array_size[dim]     = (id == 0) ? (half_n) : (n-half_n);
      array_sub_size[dim] = array_size[dim];
    }

    for (std::size_t dim = 0; dim < NumDim; ++dim)
    {
      auto temp_sub   = array_sub_size[dim];
      auto temp_size  = array_size[dim];

      array_sub_size[dim] = 1;
      array_size[dim]     = DD.get_topology().__local_shape[dim] * 2;

      MPI_Type_create_subarray( DD.get_topology().__dimension,
                                array_size, array_sub_size, array_starts,
                                MPI_ORDER_C, DD.get_topology().__mpi_value_type, &Halos[dim]);

      MPI_Type_commit(&Halos[dim]);

      array_sub_size[dim] = temp_sub;
      array_size[dim]     = temp_size;
    }
    
    for (std::size_t i = 0; i < 1000; ++i)
    {
      #pragma omp reduction(+:diff)
      {
        diff = update_omp2(DD, id);
      }

      #pragma omp single
      {
        MPI_Reduce(&diff, &gdiff, 1, MPI_DOUBLE, MPI_SUM, 0, world.comm());
        // if (world.rank() == 0 && i % 1000 == 0) 
        // std::cout << std::fixed << std::setw(20) << std::setprecision(15) << gdiff << std::endl;
        
      }

      #pragma omp single
      {
        if ( i > 998 ) 
        {
          std::cout << DD.get_array() << std::endl;
        }
      }

      #pragma omp critical
      {
        std::size_t dim {0};
        auto flag {dim};
        auto n_size {DD.get_array().__local_array.__shape[dim]};

        MPI_Sendrecv( &DD.get_array().__local_array(1,1+id), 1, 
                      Halos[dim], DD.get_topology().__neighbors[2*dim], flag,
                      &DD.get_array().__local_array(n_size-1, 1+id), 1, 
                      Halos[dim], DD.get_topology().__neighbors[2*dim+1], flag,
                      DD.get_topology().__comm_cart, MPI_STATUS_IGNORE);

        MPI_Sendrecv( &DD.get_array().__local_array(n_size-2,1+id), 1, 
                      Halos[dim], DD.get_topology().__neighbors[2*dim+1], flag,
                      &DD.get_array().__local_array(0, 1+id), 1, 
                      Halos[dim], DD.get_topology().__neighbors[2*dim], flag,
                      DD.get_topology().__comm_cart, MPI_STATUS_IGNORE);

        dim = 1;
        flag = dim;
        n_size = DD.get_array().__local_array.__shape[dim];
        MPI_Sendrecv( &DD.get_array().__local_array(1+id,1), 1, 
                      Halos[dim], DD.get_topology().__neighbors[2*dim], flag,
                      &DD.get_array().__local_array(1+id, n_size-1), 1, 
                      Halos[dim], DD.get_topology().__neighbors[2*dim+1], flag,
                      DD.get_topology().__comm_cart, MPI_STATUS_IGNORE);

        MPI_Sendrecv( &DD.get_array().__local_array(1+id, n_size-2), 1, 
                      Halos[dim], DD.get_topology().__neighbors[2*dim+1], flag,
                      &DD.get_array().__local_array(1+id, 0), 1, 
                      Halos[dim], DD.get_topology().__neighbors[2*dim], flag,
                      DD.get_topology().__comm_cart, MPI_STATUS_IGNORE);

                if ( i > 998 )
                {
                  std::size_t dim {0};
                  auto n_size {DD.get_array().__local_array.__shape[dim]};
                  int tsize {0};
                  MPI_Type_size(Halos[dim], &tsize);

                  std::cout 
                  << "PROC: " << DD.get_topology().__rank << "/" << DD.get_topology().__num_procs 
                  << " T: " << id << "/" << 2
                  << " SRs in dim : " << dim << "/" << DD.get_topology().__dimension 
                  << " SEND " << tsize / 8 << " Ds at " <<              &DD.get_array().__local_array(1,1+id) 
                    << " to " << DD.get_topology().__neighbors[2*dim] 
                  << " RECV " <<                                        &DD.get_array().__local_array(n_size-1, 1+id) 
                    << " from " << DD.get_topology().__neighbors[2*dim + 1]
                  << " \t "  
                  << " SEND " <<                                        &DD.get_array().__local_array(n_size-2, 1+id) 
                    << " to " << DD.get_topology().__neighbors[2*dim + 1] 
                  << " RECV " <<                                        &DD.get_array().__local_array(0,1+id) 
                    << " from " << DD.get_topology().__neighbors[2*dim]
                  << std::endl;

                  dim = 1;
                  n_size = DD.get_array().__local_array.__shape[dim];
                  MPI_Type_size(Halos[dim], &tsize);
                  std::cout 
                  << "PROC: " << DD.get_topology().__rank << "/" << DD.get_topology().__num_procs 
                  << " T: " << id << "/" << 2
                  << " SRs in dim : " << dim << "/" << DD.get_topology().__dimension 
                  << " SEND " << tsize / 8 << " Ds at " <<              &DD.get_array().__local_array(1+id,1) 
                    << " to " << DD.get_topology().__neighbors[2*dim] 
                  << " RECV " <<                                        &DD.get_array().__local_array(1+id,n_size-1) 
                    << " from " << DD.get_topology().__neighbors[2*dim + 1]
                  << " \t "  
                  << " SEND " <<                                        &DD.get_array().__local_array(1+id, n_size-2) 
                    << " to " << DD.get_topology().__neighbors[2*dim + 1] 
                  << " RECV " <<                                        &DD.get_array().__local_array(1+id,0) 
                    << " from " << DD.get_topology().__neighbors[2*dim]
                  << std::endl;

                  std::cout << "\n";
                }
      }

    }


    for (std::size_t dim = 0; dim < NumDim; ++dim) MPI_Type_free(&Halos[dim]);
  }
  double t2 {MPI_Wtime() - t1};

  Gather(G, DD);
  // if (world.rank() == 0) std::cout << G.get_array() << std::endl;
  if (world.rank() == 0) G.saveToBinaryFile("TEST.bin");
  MPI_Reduce(&t2, &t_com, 1, MPI_DOUBLE, MPI_SUM, 0, world.comm());
  // if (world.rank() == 0) std::cout << t_com << std::endl;

  
  return 0;
}