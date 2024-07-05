#include <iostream>

#include <cmath>

#include <omp.h>
#include <vector>
#include <cstring>

#include "update.hpp"


int main( int argc, char ** argv)
{
  auto world {final_project::mpi::env(argc, argv)};

  auto shape {final_project::__detail::__types::__multi_array_shape<2>(190, 170)};

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

    auto nx {DD.get_topology().__local_shape[0] - 2};
    auto ny {DD.get_topology().__local_shape[1] - 2};

    auto nx_h {static_cast<int>(std::round(static_cast<double>(nx) / 2.0))};
    auto ny_h {static_cast<int>(std::round(static_cast<double>(ny) / 2.0))};

    auto nx_cnt {(id == 0) ? (nx_h) : (nx-nx_h)};
    auto ny_cnt {(id == 0) ? (ny_h) : (ny-ny_h)};

    MPI_Type_vector(ny_cnt, 1, 2,         DD.get_topology().__mpi_value_type, &Halos[0]);
    MPI_Type_vector(nx_cnt, 1, 2*(ny+2),  DD.get_topology().__mpi_value_type, &Halos[1]);

    MPI_Type_commit(&Halos[0]);
    MPI_Type_commit(&Halos[1]);
    
    for (std::size_t i = 0; i < 235219; ++i)
    {
      #pragma omp reduction(+:diff)
      {
        diff = update_omp2(DD, id);
      }

      // #pragma omp critical
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