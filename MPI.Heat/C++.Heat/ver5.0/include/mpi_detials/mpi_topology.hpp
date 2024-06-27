///
/// @file topology.hpp
/// @brief This file contains the details of types design, specifically 
/// for MPI Topology structure.
///
/// @author LI Yihai
/// @version 5.0 
/// @date Jun 26, 2024
///

#pragma once 
#include <memory>
#include "multi_array/types.hpp"
#include "mpi_types.hpp"
#include "mpi_environment.hpp"


// ------------------------------- Header File ------------------------------- // 

namespace final_project {
namespace __detail {
namespace __mpi_types {

typedef std::size_t __size_type;

/// @brief Represents the topology information for MPI Cartesian 
/// Topology structure which is aiming for Array distributing.
/// @tparam _T The data type of elements.
/// @tparam _NumDims The type used for dimensions and sizes.
template <typename __T, __size_type __NumD>
  struct __mpi_topology {

  // Using Datatypes
  public:
  typedef __types::__multi_array_shape<__NumD> __super_array_shape;
  typedef final_project::mpi::env              __mpi_env;

  // Global Features
  public:
  int __dimension {__NumD}, __num_procs;
  __super_array_shape __global_shape;

  MPI_Datatype __mpi_value_type;
  MPI_Comm __comm_cart;

  // Local Features
  public:
  __super_array_shape __local_shape {__global_shape};

  int __rank;
  int __starts[__NumD], __ends[__NumD];
  int __dims[__NumD], __periods[__NumD], __neighbors[__NumD * 2], __coordinates[__NumD];
  MPI_Datatype __halo_vectors[__NumD];

  public:
  /// @brief Default constructor of __mpi_topology
  __mpi_topology( );

  /// @brief Constructor of _mpi_topology
  /// @param __glob_shape The global of the array
  /// @param __env  The MPI environment of distributed arrays.
  __mpi_topology( __super_array_shape __glob_shape, __mpi_env& __env );

  /// @brief Destructor of __mpi_topology
  ~__mpi_topology();
}; // struct __mpi_topology



} // namespace __mpi_types
} // namespace __detail
} // namespace final_project


// ------------------------------- Source File ------------------------------- // 

namespace final_project {
namespace __detail {
namespace __mpi_types {

template <typename __T, __size_type __NumD>
  inline 
  __mpi_topology<__T, __NumD>::~__mpi_topology()
  {
    std::cout << " PROC " << __rank << " Calling destructor \n";
    // Free halo vector data types if they were created
    for (int i = 0; i < __NumD; ++i) {
      if (__halo_vectors[i] != MPI_DATATYPE_NULL) {
        MPI_Type_free(&__halo_vectors[i]);
      }
    }

      // Free the cartesian communicator if it was created
      if (__comm_cart != MPI_COMM_NULL) {
        MPI_Comm_free(&__comm_cart);
      }
    }


template <typename __T, __size_type __NumD>
  inline 
  __mpi_topology<__T, __NumD>::__mpi_topology()
  : __dimension(__NumD), __num_procs(0), __global_shape(),
    __mpi_value_type(MPI_DATATYPE_NULL), __comm_cart(MPI_COMM_NULL), 
    __local_shape(__global_shape), __rank(0)
  {
    std::fill(      std::begin(__starts), std::end(__starts),       0);
    std::fill(        std::begin(__ends), std::end(__ends),         0);
    std::fill(        std::begin(__dims), std::end(__dims),         0);
    std::fill(     std::begin(__periods), std::end(__periods),      0);
    std::fill(   std::begin(__neighbors), std::end(__neighbors),    0);
    std::fill( std::begin(__coordinates), std::end(__coordinates),  0);
    std::fill(std::begin(__halo_vectors), std::end(__halo_vectors), MPI_DATATYPE_NULL);
  }

template <typename __T, __size_type __NumD>
  inline 
  __mpi_topology<__T, __NumD>::__mpi_topology( __super_array_shape __global_shape, __mpi_env& __env)
  : __global_shape {__global_shape}
  {
    // Global Features
    __mpi_value_type = __get_mpi_type<__T>();

    for (__size_type i = 0; i < __NumD; ++i) { __dims[i] = 0; }
    MPI_Dims_create(__env.size(), __dimension, __dims);
    
    MPI_Cart_create(__env.comm(), __dimension, __dims, __periods, 1, &__comm_cart);
    MPI_Comm_size(__comm_cart, &__num_procs);

    // Local Features 
    MPI_Comm_rank(__comm_cart, &__rank);
    MPI_Cart_coords(__comm_cart, __rank, __dimension, __coordinates);


auto decomp = [](const int n, const int prob_size, const int rank, int& s, int& e)
{
  int n_loc {n / prob_size}, deficit {n % prob_size};

  s = rank * n_loc + 1;
  s += ((rank < deficit) ? rank : deficit);

  if (rank < deficit) ++n_loc;
  e = s + n_loc - 1;

  if (e > n || rank == prob_size - 1) e = n;
  
  return 0;
};


    int __array_sizes[__dimension], __array_sub_sizes[__dimension], __array_starts[__dimension];
    for (__size_type i = 0; i < __NumD; ++i)
    {
MPI_Cart_shift(__comm_cart, i, 1, &(__neighbors[2*i]), &(__neighbors[2*i+1]));

decomp(__global_shape[i]-2, __dims[i], __coordinates[i], __starts[i], __ends[i]);
__local_shape[i] = __ends[i] - __starts[i] + 1 + 2;

__array_starts[i]    = 0;
__array_sizes[i]     = __local_shape[i];
__array_sub_sizes[i] = __array_sizes[i] - 2;
    }

    for (__size_type i = 0; i < __NumD; ++i)
    {
__array_sub_sizes[i] = 1;
MPI_Type_create_subarray(__dimension, __array_sizes, __array_sub_sizes, __array_starts, 
                          MPI_ORDER_C, __mpi_value_type, &__halo_vectors[i]);
MPI_Type_commit(&__halo_vectors[i]);
    }



  }


} // namespace __mpi_types
} // namespace __detail
} // namespace final_project