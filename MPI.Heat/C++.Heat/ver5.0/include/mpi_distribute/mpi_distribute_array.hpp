///
/// @file mpi_distribute_array
/// @brief This file includes the distributed X Dimension array 
///       with defined MPI Cartesian Topology structure in file 
///       mpi_topology.hpp.
///
/// @author LI Yihai
/// @version 5.0
/// @date Jun 26, 2024

#pragma once

#include "multi_array/array.hpp"

#include "mpi_detials/mpi_topology.hpp"

#include "assert.hpp"

#ifndef FINAL_PROJECT_MPI_DISTRIBUTE_ARRAY_HPP_LIYIHAI
#define FINAL_PROJECT_MPI_DISTRIBUTE_ARRAY_HPP_LIYIHAI


// __attribute__((visibility("default")))
namespace final_project {
namespace __detail {

typedef __types::__size_type __size_type;

template <class __T, __size_type __NumD>
  class __mpi_distribute_array {
    public:
    typedef mpi::env                                 __mpi_env;

    typedef __multi_array::__array<__T, __NumD>      __array;
    typedef __types::__multi_array_shape<__NumD>     __super_array_shape;
    typedef __mpi_types::__mpi_topology<__T, __NumD> __topology;


    public:
    __array     __local_array;
    __topology  __local_topology;

    public:    
    __mpi_distribute_array( __super_array_shape  __global_shape , __mpi_env& __env ) 
    : __local_topology(__global_shape, __env),
      __local_array(__local_topology.__local_shape)
    {
      std::cout 
      << "PROCS " << __local_topology.__rank 
      << " constructs distributed " << __local_topology.__dimension << "D Array:"
      << " Global Shape [" 
      << __global_shape[0] << ", " 
      << __global_shape[1] << ", " 
      << __global_shape[2] << "]" 
      << " | Local Shape ["
      << __local_topology.__local_shape[0] << ", "
      << __local_topology.__local_shape[1] << ", "
      << __local_topology.__local_shape[2] << "]" 
      << " | Coords " << "[("
      << __local_topology.__starts[0] << ", " << __local_topology.__ends[0] << "), ("
      << __local_topology.__starts[1] << ", " << __local_topology.__ends[1] << "), ("
      << __local_topology.__starts[2] << ", " << __local_topology.__ends[2] << ")] "
      << std::endl;
    }




  }; // __mpi_distribute_array
  
} // namespace __detail
} // namespace final_project










#endif // FINAL_PROJECT_MPI_DISTRIBUTE_ARRAY_HPP_LIYIHAI