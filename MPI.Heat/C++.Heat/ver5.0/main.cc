#include <iostream>
// #include "mpi_detials/mpi_topology.hpp"
// #include "mpi_detials/mpi_types.hpp"
// #include "mpi_detials/mpi_environment.hpp"
// #include "multi_array/base.hpp"
// #include "mpi_distribute/mpi_distribute_array.hpp"
// #include "array.hpp"

#include <cmath>

#include <omp.h>
#include <vector>
#include <cstring>

#include "update.hpp"


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


int main( int argc, char ** argv)
{
  auto world {final_project::mpi::env(argc, argv)};


  auto shape {final_project::__detail::__types::__multi_array_shape<2>(500, 500)};
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
  DD.fill_boundary(10);
  GG.fill_boundary(10);

  // std::cout << DD.get_array() << std::endl;
  double diff {0}, gdiff {0};

  auto t1 = MPI_Wtime();
  for (std::size_t i = 0; i < 1000000; ++i)
  {
    // diff = update_ping_pong1(DD, GG);
    // diff = update_ping_pong1(GG, DD);

    diff = update_omp1(DD);

    MPI_Allreduce(&diff, &gdiff, 1, MPI_DOUBLE, MPI_SUM, world.comm());

    if (gdiff <= 1E-10) {
      std::cout << "Converge at : " << i << std::endl;
      break;
    }

    if (world.rank() == 0 && i % 1000 == 0) 
    std::cout << std::fixed << std::setprecision(15) << std::setw(15) << gdiff << std::endl;
    
  }
  auto t2 = MPI_Wtime();
  // update(GG, DD);
  // std::cout << DD.get_array() << std::endl;

  // MPI_Barrier(world.comm());
  // exchange(DD);
  // MPI_Barrier(world.comm());
  // std::cout << DD.get_array() << std::endl;
  auto G {final_project::array::array_base<double,2>(shape)};

  Gather(G, DD);
  // if (world.rank() == 0) std::cout << G.get_array() << std::endl;
  if (world.rank() == 0) G.saveToBinaryFile("TEST.bin");

  std::cout << t2 - t1 << std::endl;
  
  // std::cout 
  // << "PROC : " << DD.get_topology().__rank 
  // << " (" << DD.get_topology().__rank << ",1) >>> " 
  // << DD.get_array().__local_array(DD.get_topology().__rank,1) 
  // << std::endl;
  // 388175
  // 406114
  
  

  return 0;
}