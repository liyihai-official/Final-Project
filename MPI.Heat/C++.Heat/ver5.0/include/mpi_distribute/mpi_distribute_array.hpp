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

#include "multi_array/base.hpp"

#include "mpi_detials/mpi_topology.hpp"

#include "assert.hpp"

#include <unistd.h>

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
    __topology  __local_topology;
    __array     __local_array;

    public:    
    __mpi_distribute_array( __super_array_shape & __global_shape , __mpi_env& __env ) 
    : __local_topology(__global_shape, __env),
      __local_array(__local_topology.__local_shape)
    {
      
      std::cout 
      << "PROCS " << std::fixed << std::setw(2) << __local_topology.__rank 
      << " constructs distributed " << __local_topology.__dimension << "D Array:"
      << " Global Shape [" 
      << __global_shape[0] 
      << ", " << __global_shape[1] 
      // << ", "  << __global_shape[2] 
      << "]" 
      << " | Local Shape ["
      << __local_topology.__local_shape[0] << ", "
      << __local_topology.__local_shape[1] << ", "
      // << __local_topology.__local_shape[2] 
      << "]" 
      << " | Coords " << "[("
      << std::fixed << std::setw(4) << __local_topology.__starts[0] << ", " 
      << std::fixed << std::setw(4) << __local_topology.__ends[0] << "), ("
      << std::fixed << std::setw(4) << __local_topology.__starts[1] << ", " 
      << std::fixed << std::setw(4) << __local_topology.__ends[1] << ")" 
      // << ", (" << __local_topology.__starts[2] << ", " << __local_topology.__ends[2] 
      << ")] "
      << std::endl;
    }

    void swap(__mpi_distribute_array & other)
    {
      std::cout << "Swap __mpi_distributed_array " << __local_topology == other.__local_topology << std::endl;
  FINAL_PROJECT_ASSERT_MSG((__local_topology == other.__local_topology), "Matched MPI Topology required for swapping");
      __local_array.swap(other.__local_array);
    }

    /// @brief Print Distribute arrays in order of Rank
    template <class __U, __size_type __Dims>
    friend std::ostream& operator<<(std::ostream& os, const __mpi_distribute_array<__U, __Dims>& in);

  }; // __mpi_distribute_array
  
} // namespace __detail
} // namespace final_project



// ------------------------------- Source File ------------------------------- // 
namespace final_project {
namespace __detail {

template <class __U, __size_type __Dims>
std::ostream& operator<<(std::ostream& os, const __mpi_distribute_array<__U, __Dims>& in)
{
  MPI_Barrier(in.__local_topology.__comm_cart);
  sleep(1);
  os << "Attempting to print array in order \n";
  MPI_Barrier(in.__local_topology.__comm_cart);

  for (int i = 0; i < in.__local_topology.__num_procs; ++i)
  {
    if ( i == in.__local_topology.__rank )
    {
  os
  << "\nPROC : " << in.__local_topology.__rank << " of " 
  << in.__local_topology.__num_procs <<  " is Printing \n" 
  << in.__local_array;
    }
    fflush(stdout);
    sleep(0.1);
    MPI_Barrier(in.__local_topology.__comm_cart);
  }

  return os;
}



} // namespace __detail
} // namespace final_project


#endif // FINAL_PROJECT_MPI_DISTRIBUTE_ARRAY_HPP_LIYIHAI