/**
 * @file base.cpp
 * @brief This file contains the definition of the array classes
 *        and features for parallel processing.
 * 
 * The classes provided in this file are designed to set up basic 
 * features of a skeleton 1D, 2D and 3D array. And enhanced them into
 * distributed 1D, 2D and 3D array used in parallel processing 
 * environments.
 * 
 * 
 * @author LI Yihai
 * @version 4.0
 * @date Jun 6, 2024
 */

#ifndef FINAL_PROJECT_MULTI_BASE_HPP_LIYIHAI
#define FINAL_PROJECT_MULTI_BASE_HPP_LIYIHAI

#include "types"
#include "multi_types.cpp"
#include <memory>


namespace final_project {
namespace _detail {
namespace _multi_array {

template <class _T, _size_type _NumDim>
class _array {
  public:
    typedef _T         value_type;
    typedef _T&        reference;
    typedef const _T&  const_reference;
    typedef _T*        iterator;
    typedef const _T*  const_iterator;

    typedef final_project::_detail::_size_type            _size_type;
    typedef final_project::_detail::_multi_array_shape<_NumDim> _shape_type;
  
  public:
    _shape_type           _shape;
    std::unique_ptr<_T[]>  _data;
  
  public:
    _array(_shape_type _shape) 
      : _shape {_shape}, _data {std::make_unique<_T[]>(_shape.size())} { }

  public:
    // Sizes
    const _size_type size()       { return _shape.size(); }
    const _size_type size() const { return _shape.size(); }

    // Iterators
    iterator          begin()       { return _data.get(); }
    const_iterator    begin() const { return _data.get(); }
    const_iterator   cbegin() const { return _data.get(); }

    iterator            end()       { return _data.get() + this->size(); }
    const_iterator      end() const { return _data.get() + this->size(); }
    const_iterator     cend() const { return _data.get() + this->size(); }

    // Operators ()
    template <typename... Args>
    reference operator()(Args... args) {
FINAL_PROJECT_ASSERT_MSG((sizeof...(args) == _NumDim), "Number of arguments must match the number of dimensions.");

      _size_type indices[] = { static_cast<_size_type>(args)... };
      _size_type index = 0;
      _size_type multiplier = 1;

      for (_size_type i = 0; i < _NumDim; ++i) {
          index += indices[_NumDim - 1 - i] * multiplier;
          multiplier *= _shape[_NumDim - 1 - i];
      }

      return _data[index];
    }

    reference operator[](_size_type index) { 
FINAL_PROJECT_ASSERT_MSG((index < _shape.size()), "Index out of range.");
      return _data[index]; 
    }

  public:
    void fill (const_reference value)   { std::fill_n(begin(), size(), value); }
    void assign (const_reference value) { fill (value); }

}; // class array 1d







} // namespace _multi_array
} // namespace _detail
} // namespace final_project

#endif /* FINAL_PROJECT_ARRAY_HPP */