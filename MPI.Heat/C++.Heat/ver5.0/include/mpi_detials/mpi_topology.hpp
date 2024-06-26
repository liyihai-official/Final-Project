///
/// @file topology.hpp
/// @brief This file contains the details of types design, specifically 
/// for MPI Topology structure.
///
/// @author LI Yihai
/// @version 5.0 
/// @date Jun 26, 2024

#pragma once 
#include <memory>
#include "multi_array/types.hpp"
#include "mpi_types.hpp"
#include "mpi_environment.hpp"


// ------------------------------- Header File ------------------------------- // 

namespace final_project {
namespace __detail {
namespace __mpi_types {

typedef std::size_t _size_type;

/// @brief Represents the topology information for MPI Cartesian 
/// Topology structure which is aiming for Array distributing.
/// @tparam _T The data type of elements.
/// @tparam _NumDims The type used for dimensions and sizes.
template <typename _T, _size_type _NumD>
  struct __mpi_topology {

  // Using Datatypes
  public:
  typedef __types::_multi_array_shape<_NumD> _super_array_shape;
  typedef final_project::mpi::env            _mpi_env;

  // Global Features
  public:
  int _dimension {_NumD}, _num_procs;
  _super_array_shape _global_shape;

  MPI_Datatype _mpi_value_type;
  MPI_Comm _comm_cart;

  // Local Features
  public:
  _super_array_shape _local_shape {_global_shape};

  int _rank;
  int _starts[_NumD], _ends[_NumD];
  int _dims[_NumD], _periods[_NumD], _neighbors[_NumD * 2], _coordinates[_NumD];
  MPI_Datatype _halo_vectors[_NumD];

  public:
  /// @brief Constructor of _mpi_topology
  /// @param _glob_shape The global of the array
  /// @param _env  The MPI environment of distributed arrays.
  __mpi_topology( _super_array_shape _glob_shape, _mpi_env& _env );

}; // struct __mpi_topology



} // namespace __mpi_types
} // namespace __detail
} // namespace final_project


// ------------------------------- Source File ------------------------------- // 

namespace final_project {
namespace __detail {
namespace __mpi_types {


template <typename _T, _size_type _NumD>
  inline 
  __mpi_topology<_T, _NumD>::__mpi_topology( _super_array_shape _global_shape, _mpi_env& _env)
  : _global_shape {_global_shape}
  {
    // Global Features
    _mpi_value_type = _get_mpi_type<_T>();

    for (_size_type i = 0; i < _NumD; ++i) { _dims[i] = 0; }
    MPI_Dims_create(_env.size(), _dimension, _dims);
    
    MPI_Cart_create(_env.comm(), _dimension, _dims, _periods, 1, &_comm_cart);
    MPI_Comm_size(_comm_cart, &_num_procs);

    // Local Features 
    MPI_Comm_rank(_comm_cart, &_rank);
    MPI_Cart_coords(_comm_cart, _rank, _dimension, _coordinates);


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


    int _array_sizes[_dimension], _array_sub_sizes[_dimension], _array_starts[_dimension];
    for (_size_type i = 0; i < _NumD; ++i)
    {
MPI_Cart_shift(_comm_cart, i, 1, &(_neighbors[2*i]), &(_neighbors[2*i+1]));

decomp(_global_shape[i]-2, _dims[i], _coordinates[i], _starts[i], _ends[i]);
_local_shape[i] = _ends[i] - _starts[i] + 1 + 2;

_array_starts[i]    = 0;
_array_sizes[i]     = _local_shape[i];
_array_sub_sizes[i] = _array_sizes[i] - 2;
    }

    for (_size_type i = 0; i < _NumD; ++i)
    {
_array_sub_sizes[i] = 1;
MPI_Type_create_subarray(_dimension, _array_sizes, _array_sub_sizes, _array_starts, 
                          MPI_ORDER_C, _mpi_value_type, &_halo_vectors[i]);
MPI_Type_commit(&_halo_vectors[i]);
    }



  }


} // namespace __mpi_types
} // namespace __detail
} // namespace final_project