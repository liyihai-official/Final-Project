#include <iostream>
#include "mpi_detials/mpi_topology.hpp"
#include "mpi_detials/mpi_types.hpp"
#include "mpi_detials/mpi_environment.hpp"
#include "multi_array/array.hpp"
#include "mpi_distribute/mpi_distribute_array.hpp"



int main( int argc, char ** argv)
{
  auto world {final_project::mpi::env(argc, argv)};


  auto shape {final_project::__detail::__types::__multi_array_shape<2>(6, 5)};
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


  auto DA {final_project::__detail::__mpi_distribute_array<double, 2>(shape, world)};
  DA.__local_array.fill(0);

  // if (world.rank() == 0) 
  // {
    std::cout << DA << std::endl;
  // }


  return 0;
}