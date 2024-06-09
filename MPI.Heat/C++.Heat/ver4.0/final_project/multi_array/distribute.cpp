#pragma once 
#include "base.cpp"
#include "final_project/environment.cpp"

namespace final_project {
namespace _detail {
namespace _multi_array {

template <class _T, _detail::_types::_size_type _NumDims>
  class _array_distribute
  {
    public:
      typedef _T          _value_type;
      typedef const _T    _const_value_type;

      typedef _T&         _reference;
      typedef const _T&   _const_reference;

      typedef _T*         _iterator;
      typedef const _T*   _const_iterator;

      typedef _detail::_types::_size_type                   _size_type;
      typedef _detail::_types::_multi_array_shape<_NumDims> _super_array_shape;

      typedef final_project::mpi::env                       _mpi_env;
      typedef MPI_Datatype                                  _mpi_value_type;
      typedef _detail::_types::_mpi_topology<_T, _NumDims>  _topology;

    public:
      std::unique_ptr<_array<_T, _NumDims>> _distr_array;

      _topology _mpi_topo;

    public:
      // sizes 
      _size_type size()       { return _mpi_topo._loc_shape.size(); }
      _size_type size() const { return _mpi_topo._loc_shape.size(); }
      _super_array_shape shape()      const { return _mpi_topo._loc_shape;  } 
      _super_array_shape glob_shape() const { return _mpi_topo._glob_shape; }


      template <typename ... Args>
      _reference operator()(Args ... args)     { return (*_distr_array)(args...); }
      _reference operator[](_size_type index)  { return (*_distr_array)[index];   }


    public:
      _array_distribute(_super_array_shape _glob_shape, _mpi_env& _env)
       : _mpi_topo(_glob_shape, _env)
      { 
        _distr_array = std::make_unique<_array<_T, _NumDims>>(_mpi_topo._loc_shape);
      }
      
    public:
      void fill (_const_reference value) { _distr_array->fill(value); }

      template <class _U, _size_type _Dims>
      friend std::ostream& operator<<(std::ostream& os, const _array_distribute<_U, _Dims>& in)
      {
        os << *(in._distr_array);
        return os;
      }

  };


}
}
}



