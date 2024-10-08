///
/// @file topology.cpp
/// @brief This file contains the details of types design, specifically 
/// for MPI Topology structure.
///
/// @version 4.1 Jun 7


namespace final_project {
namespace _detail {
namespace _types {

/// @brief Retrieve the MPI datatype for a given template type.
///
/// @tparam T the data type.
/// @return MPI_Datatype The MPI type correspond for T data type.
template<typename T>
MPI_Datatype _get_mpi_type();


/// @brief Represents the topology information for MPI Cartesian 
/// Topology structure which is aiming for Array distributing.
/// @tparam _T The data type of elements.
/// @tparam _NumDims The type used for dimensions and sizes.
template <typename _T, _size_type _NumDims>
struct _mpi_topology
{
  public:
  typedef _multi_array_shape<_NumDims> _super_array_shape; 
  typedef final_project::mpi::env      _mpi_env;

  public:
  _super_array_shape  _glob_shape;
  _super_array_shape  _loc_shape {_glob_shape};

  int _dimension {_NumDims};
  int _rank, _size;
  std::unique_ptr<int[]> _starts, _ends;
  std::unique_ptr<int[]> _dims, _periods, _neighbors, _coordinates;
  
  MPI_Comm _comm_cart;
  MPI_Datatype  _type;
  MPI_Datatype  _vecs[_NumDims];
    
  public:
  /// @brief Constructor of _mpi_topology
  /// @param _glob_shape The global of the array
  /// @param _env  The MPI environment of distributed arrays.
  _mpi_topology( _super_array_shape _glob_shape, _mpi_env& _env );

};

} // namespace _types
} // namespace _detail
} // namespace final_project

#include "final_project/types"
#include <mpi.h>

namespace final_project {
namespace _detail {
namespace _types {

template<>
MPI_Datatype _get_mpi_type<int>()    { return MPI_INT; }
 
template<>
MPI_Datatype _get_mpi_type<float>()  { return MPI_FLOAT; }

template<>
MPI_Datatype _get_mpi_type<double>() { return MPI_DOUBLE; }

template <typename _T, _size_type _NumDims>
  inline
  _mpi_topology<_T, _NumDims>::_mpi_topology(_super_array_shape _glob_shape, _mpi_env& _env)
  : _glob_shape {_glob_shape}
  {
    _type    = _types::_get_mpi_type<_T>();

    _dims    = std::make_unique<int[]>(_dimension);
    _periods = std::make_unique<int[]>(_dimension);
    _starts  = std::make_unique<int[]>(_dimension);
    _ends    = std::make_unique<int[]>(_dimension);
    _coordinates = std::make_unique<int[]>(    _dimension);
    _neighbors   = std::make_unique<int[]>(2 * _dimension);

    MPI_Dims_create(_env.size(), _dimension, _dims.get());
    MPI_Cart_create(_env.comm(), _dimension, _dims.get(), _periods.get(), 1, &_comm_cart);
    
    MPI_Comm_rank(_comm_cart, &_rank);
    MPI_Comm_size(_comm_cart, &_size);
    MPI_Cart_coords(_comm_cart, _rank, _dimension, _coordinates.get());

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

    int _array_of_sizes[_dimension], _array_of_sub_sizes[_dimension], _array_of_starts[_dimension];

    for (_size_type i = 0; i < _NumDims; ++i)
    {
MPI_Cart_shift(_comm_cart, i, 1, &(_neighbors[2*i]), &(_neighbors[2*i+1]));

decomp(_glob_shape[i]-2, _dims[i], _coordinates[i], _starts[i], _ends[i]);
_loc_shape[i] = _ends[i] - _starts[i] + 1 + 2; 

_array_of_sizes[i]  = _loc_shape[i];
_array_of_starts[i] = 0;

_array_of_sub_sizes[i] = _array_of_sizes[i] - 2;
    }

    for (_size_type i = 0; i < _NumDims; ++i)
    {
_array_of_sub_sizes[i] = 1;
MPI_Type_create_subarray(_dimension, _array_of_sizes, _array_of_sub_sizes, _array_of_starts, 
                        MPI_ORDER_C, _type, &_vecs[i]);

MPI_Type_commit(&_vecs[i]);
    }

  }


} // namespace _types
} // namespace _detail
} // namespace final_project