///
/// @file types.hpp
/// @brief This file is the header file of customized datatypes for 
///         multi-dimension arrays.
/// 
/// @author LI Yihai
/// @version 5.0 
/// @date Jun 26, 2024
///

#ifndef FINAL_PROJECT_TYPES_HPP_LIYIHAI
#define FINAL_PROJECT_TYPES_HPP_LIYIHAI

// ------------------------------- Header File ------------------------------- // 

#include <memory>

namespace final_project {
namespace __detail {
namespace __types {

typedef std::size_t __size_type;

/// @brief 
/// @tparam __NumD 
template <__size_type __NumD>
  struct __multi_array_shape {

  __size_type sizes[__NumD];

  // Operators
        __size_type& operator[] (__size_type index);
  const __size_type& operator[] (__size_type index) const;

  // Member Functions
  __size_type num_dim()                     const;
  __size_type num_size()                    const;
  __size_type num_size( __size_type index ) const;

  bool is_in(__size_type index)            const;

  // Constructors
  template <typename ... Args>
  __multi_array_shape( Args ... args );

  __multi_array_shape(const __multi_array_shape& other);  

}; // struct _multi_array_shape
  

} // namespace __types
} // namespace __detail
} // namespace final_project




// ------------------------------- Source File ------------------------------- // 
#include "assert.hpp"

namespace final_project {
namespace __detail {
namespace __types {


template <__size_type __NumD>
template <typename ... Args>
  inline 
  __multi_array_shape<__NumD>::__multi_array_shape(Args ... args) 
  {
FINAL_PROJECT_ASSERT_MSG((sizeof...(args) == __NumD), "Number of arguments must match the number of dimensions.");
__size_type temp[] = { static_cast<__size_type>(args)... };
std::copy(temp, temp + __NumD, sizes);
  }

template <__size_type __NumD>
  inline 
  __multi_array_shape<__NumD>::__multi_array_shape(const __multi_array_shape& other)
  {
std::copy(other.sizes, other.sizes + __NumD, sizes);
  }

template <__size_type __NumD>
  inline 
  __size_type& __multi_array_shape<__NumD>::operator[] (__size_type index) 
  { return sizes[index]; }

template <__size_type __NumD>
  inline 
  const __size_type& __multi_array_shape<__NumD>::operator[] (__size_type index) const
  { return sizes[index]; }

template <__size_type __NumD>
  inline 
  __size_type __multi_array_shape<__NumD>::num_dim() const 
  { return __NumD; }

template <__size_type __NumD>
  inline 
  __size_type __multi_array_shape<__NumD>::num_size() const
  {
__size_type total {1};
for (__size_type i = 0; i < __NumD; ++i) total *= sizes[i];
return total;
  }

template <__size_type __NumD>
  inline
  __size_type __multi_array_shape<__NumD>::num_size(__size_type index) const 
  { return sizes[index]; }

template <__size_type __NumD>
  inline 
  bool __multi_array_shape<__NumD>::is_in( __size_type index ) const 
  {  return index < this->num_size(); }





} // namespace __types  
} // namespace __detail
} // namespace final_project



#endif