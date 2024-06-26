/// @file types.hpp
/// @brief This file is the header file of customized datatypes for 
///         multi-dimension arrays.
/// 
/// @author LI Yihai
/// @version 5.0 
/// @date Jun 26, 2024

#ifndef FINAL_PROJECT_TYPES_HPP_LIYIHAI
#define FINAL_PROJECT_TYPES_HPP_LIYIHAI

// ------------------------------- Header File ------------------------------- // 

#include <memory>

namespace final_project {
namespace __detail {
namespace __types {

typedef std::size_t _size_type;

/// 
///
template <_size_type _NumD>
  struct _multi_array_shape {

  _size_type sizes[_NumD];

  // Operators
        _size_type& operator[] (_size_type index);
  const _size_type& operator[] (_size_type index) const;

  // Member Functions
  _size_type num_dim()                    const;
  _size_type num_size()                   const;
  _size_type num_size( _size_type index ) const;

  bool is_in(_size_type index)            const;

  // Constructors
  template <typename ... Args>
  _multi_array_shape( Args ... args );

  _multi_array_shape(const _multi_array_shape& other);  

}; // struct _multi_array_shape
  

} // namespace __types
} // namespace __detail
} // namespace final_project




// ------------------------------- Source File ------------------------------- // 
#include "assert.hpp"

namespace final_project {
namespace __detail {
namespace __types {


template <_size_type _NumD>
template <typename ... Args>
  inline 
  _multi_array_shape<_NumD>::_multi_array_shape(Args ... args) 
  {
FINAL_PROJECT_ASSERT_MSG((sizeof...(args) == _NumD), "Number of arguments must match the number of dimensions.");
_size_type temp[] = { static_cast<_size_type>(args)... };
std::copy(temp, temp + _NumD, sizes);
  }

template <_size_type _NumD>
  inline 
  _multi_array_shape<_NumD>::_multi_array_shape(const _multi_array_shape& other)
  {
std::copy(other.sizes, other.sizes + _NumD, sizes);
  }

template <_size_type _NumD>
  inline 
  _size_type& _multi_array_shape<_NumD>::operator[] (_size_type index) 
  { return sizes[index]; }

template <_size_type _NumD>
  inline 
  const _size_type& _multi_array_shape<_NumD>::operator[] (_size_type index) const
  { return sizes[index]; }

template <_size_type _NumD>
  inline 
  _size_type _multi_array_shape<_NumD>::num_dim() const 
  { return _NumD; }

template <_size_type _NumD>
  inline 
  _size_type _multi_array_shape<_NumD>::num_size() const
  {
_size_type total {1};
for (_size_type i = 0; i < _NumD; ++i) total *= sizes[i];
return total;
  }

template <_size_type _NumD>
  inline
  _size_type _multi_array_shape<_NumD>::num_size(_size_type index) const 
  { return sizes[index]; }

template <_size_type _NumD>
  inline 
  bool _multi_array_shape<_NumD>::is_in( _size_type index ) const 
  {  return index < this->num_size(); }





} // namespace __types  
} // namespace __detail
} // namespace final_project



#endif