#include "base.cpp"
#include "environment.cpp"

namespace final_project {
namespace _detail {
namespace _multi_array {

  template <class _T, _detail::_types::_size_type _NumDims>
  class _array_distribute
  {
    // _array<_T, _NumDims>
    public:
      typedef _detail::_types::_size_type                   _size_type;
      typedef _detail::_types::_multi_array_shape<_NumDims> _super_array_shape;
      typedef _detail::_types::_mpi_topology<_T, _NumDims>            _topology;

      typedef MPI_Datatype              _mpi_value_type;
      typedef final_project::mpi::env   _mpi_env;

    public:
      _topology           _mpi_topo;

    public:
      _array_distribute(_super_array_shape _glob_shape, _mpi_env& _env)
       :  _mpi_topo(_glob_shape, _env) 
      { 
        
      }

  };


}
}
}



