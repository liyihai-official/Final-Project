///
/// @file topology.hpp
/// @brief This file contains the details of types design, specifically 
/// for MPI Topology structure.
///
/// @author LI Yihai
/// @version 5.0 
/// @date Jun 26, 2024

#include <memory>
#include "multi_array/types.hpp"
#include "mpi_types.hpp"


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
  struct __mpi_topology;



} // namespace __mpi_types
} // namespace __detail
} // namespace final_project



// ------------------------------- Source File ------------------------------- // 

namespace final_project {
namespace __detail {
namespace __mpi_types {

template <typename _T, _size_type _NumD>
  struct __mpi_topology {

  // Using Datatypes
  public:
  typedef __types::_multi_array_shape<_NumD> _super_array_shape;

  // Global Features
  public:
  int _dimension {_NumD}, _num_procs;
  _super_array_shape _global_shape;
  MPI_Datatype _mpi_value_type;

  // Local Features
  public:
  _super_array_shape _local_shape {_global_shape};
  std::unique_ptr<int[]> _starts, _ends;
  std::unique_ptr<int[]> _dims, _periods, _neighbors, _coordinates;
  MPI_Datatype _halo_vectors[_NumD];

  public:
  /// @brief Constructor of _mpi_topology
  /// @param _glob_shape The global of the array
  /// @param _env  The MPI environment of distributed arrays.
  _mpi_topology( _super_array_shape _glob_shape, _mpi_env& _env );

}; // struct __mpi_topology



} // namespace __mpi_types
} // namespace __detail
} // namespace final_project