/// @file distribute.cpp
/// @brief The _details of distributed array design.
/// @version 4.1
/// @date Jun. 7
#pragma once 
#include "base.cpp"
#include "final_project/environment.cpp"

namespace final_project {
namespace _detail {
namespace _multi_array {

/// @brief class of _array_distribute
/// @tparam _T The datatype of elements in array
/// @tparam _NumDims The number of dimensions of array
template <class _T, _detail::_types::_size_type _NumDims>
  class _array_distribute
  {
    /// Datatypes
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

    /// Member variables
    public:
    std::unique_ptr<_array<_T, _NumDims>> _distr_array;
    _topology _mpi_topo;

    public:
    /// @brief Constructor of _array_distribute
    /// @param _glob_shape 
    /// @param _env 
    _array_distribute(_super_array_shape _glob_shape, _mpi_env& _env)
      : _mpi_topo(_glob_shape, _env)
    { 
      _distr_array = std::make_unique<_array<_T, _NumDims>>(_mpi_topo._loc_shape);
    }


    /// Member functions
    public:
    // sizes 
    _size_type size()       { return _mpi_topo._loc_shape.size(); }
    _size_type size() const { return _mpi_topo._loc_shape.size(); }
    _super_array_shape shape()      const { return _mpi_topo._loc_shape;  } 
    _super_array_shape glob_shape() const { return _mpi_topo._glob_shape; }


    /// @brief Operator, accessing the member of array in multi-dimension 
    ///        perspective.
    /// @tparam ...Args Template of arguments
    /// @param ...args The indexes of accessing values in array.
    /// @return A value of array.
    template <typename ... Args>
    _reference operator()(Args ... args)     { return (*_distr_array)(args...); }
    
    /// @brief Operator, accessing the member of array as a single dimension
    ///        array.
    /// @param index  The index of elements.
    /// @return A value of array.
    _reference operator[](_size_type index)  { return (*_distr_array)[index];   }
      
    public:
    /// @brief Init all values in array by an input value.
    /// @param value A junk value.
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



