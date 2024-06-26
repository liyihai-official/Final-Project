#include <iostream>
#include "mpi_detials/mpi_topology.hpp"
#include "mpi_detials/mpi_types.hpp"
#include "mpi_detials/mpi_environment.hpp"


int main( int argc, char ** argv)
{
  auto world {final_project::mpi::env(argc, argv)};


  auto shape {final_project::__detail::__types::_multi_array_shape<3>(10, 9 , 8)};
  auto an_topology {final_project::__detail::__mpi_types::__mpi_topology<double, 3>(shape, world)};
  std::cout 
  << " PROCESS " << an_topology._rank 
  << " Has Coordinate : \t ["
  << an_topology._coordinates[0] << ", "
  << an_topology._coordinates[1] << ", "
  << an_topology._coordinates[2] << "]"
  << " \t "
  << " Has Shape : \t [" 
  << an_topology._local_shape[0] << ", "
  << an_topology._local_shape[1] << ", "
  << an_topology._local_shape[2] << "] "
  << " Range : \t [ ("
  << an_topology._starts[0] << ", " << an_topology._ends[0] << ") " << ", ("
  << an_topology._starts[1] << ", " << an_topology._ends[1] << ") " << ", ("
  << an_topology._starts[2] << ", " << an_topology._ends[2] << ") " << "] "
  << std::endl;

  


  return 0;
}