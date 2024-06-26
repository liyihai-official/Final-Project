#include <iostream>
#include "mpi_detials/mpi_topology.hpp"
#include "mpi_detials/mpi_types.hpp"
#include "mpi_detials/mpi_environment.hpp"
#include "multi_array/array.hpp"
#include "mpi_distribute_array.hpp"



int main( int argc, char ** argv)
{
  auto world {final_project::mpi::env(argc, argv)};


  auto shape {final_project::__detail::__types::__multi_array_shape<3>(10, 9 , 8)};
  auto an_topology {final_project::__detail::__mpi_types::__mpi_topology<double, 3>(shape, world)};
  std::cout 
  << " PROCESS " << an_topology.__rank 
  << " Has Coordinate : \t ["
  << an_topology.__coordinates[0] << ", "
  << an_topology.__coordinates[1] << ", "
  << an_topology.__coordinates[2] << "]"
  << " \t "
  << " Has Shape : \t [" 
  << an_topology.__local_shape[0] << ", "
  << an_topology.__local_shape[1] << ", "
  << an_topology.__local_shape[2] << "] "
  << " Range : \t [ ("
  << an_topology.__starts[0] << ", " << an_topology.__ends[0] << ") " << ", ("
  << an_topology.__starts[1] << ", " << an_topology.__ends[1] << ") " << ", ("
  << an_topology.__starts[2] << ", " << an_topology.__ends[2] << ") " << "] "
  << std::endl;


  auto Array {final_project::__detail::__multi_array::__array<float, 3>(shape)};
  Array.fill(0);

  if (world.rank() == 1 )
  {
    std::cout << Array << std::endl;
  }

  


  return 0;
}