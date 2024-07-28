///
/// @file   multiarray/types.hpp
/// @brief  This file defines datatypes for multi-array objects
/// @author LI Yihai
///

#ifndef FINAL_PROJECT_MULTIARRAY_TYPES_HPP_LIYIHAI
#define FINAL_PROJECT_MULTIARRAY_TYPES_HPP_LIYIHAI

#pragma once
#include <vector>       // Standard Libraries
#include <types.hpp>    // Final Project Header Files


namespace final_project { namespace multi_array {

namespace __detail {

typedef final_project::Dworld __size_type;
typedef final_project::Qworld __super_size_type;

/// @brief A shape viewer of multi-array which is __NumD dimension(s).
/// @tparam __NumD Template __size_type number, the dimensions.
template <__size_type __NumD>
  struct __multi_array_shape
  {
    // shape of array
    std::vector<__size_type> dims;

    // Operators
    __size_type& operator[] (__size_type);
    const __size_type& operator[] (__size_type) const;

    // Member Functions 
    __size_type dim() const;
    __size_type dim(__size_type) const;
    __super_size_type size() const; // Using Super for avoiding overflow

    // Constructors
    /// @brief Init an Empty dims.
    __multi_array_shape(); 

    /// @brief Construct dims from packed arguments.
    template <typename ... Args>
    __multi_array_shape( Args ... );

    /// @brief Construct dims from other one.
    __multi_array_shape(const __multi_array_shape& );

    bool operator==(__multi_array_shape &);
    bool operator!=(__multi_array_shape &);

    void swap(__multi_array_shape &);

    /// @brief A helper Function for ensuring none-negative inputs
    template <typename __T>
    __size_type check_and_cast(__T);
  };
  



} // end of namespace __detail


} // end of namespace multi_array
} // end of final_project






///
/// Definition of inline member functions
///

#include <assert.hpp>
namespace final_project { namespace multi_array {


namespace __detail { 


template <__size_type __NumD>
  inline
  __multi_array_shape<__NumD>::__multi_array_shape()
{ dims.resize(0); }

template <__size_type __NumD>
template <typename ... Args>
  inline 
  __multi_array_shape<__NumD>::__multi_array_shape(Args ... args)
{ 
  FINAL_PROJECT_ASSERT_MSG(
    (sizeof...(args) == __NumD),
    "Number of Arguments must Match the dimension."
  );

  // cast the inputs (non-negative) into __size_type (uint64_t)
  dims = { check_and_cast(args)... };
}

template <__size_type __NumD>
  inline
  __multi_array_shape<__NumD>::__multi_array_shape(
    const __multi_array_shape & other)
{ 
  dims = other.dims;
}

template <__size_type __NumD>
  inline 
  __super_size_type
  __multi_array_shape<__NumD>::size()
const
{ 
  __super_size_type total {1};
  for (auto & elem : dims) { 
    total *= static_cast<__super_size_type>(elem); 
  }
  return total;
}

template <__size_type __NumD>
  inline 
  __size_type& 
  __multi_array_shape<__NumD>::operator[] (__size_type index) 
{ return dims[index]; }

template <__size_type __NumD>
  inline 
  const __size_type& 
  __multi_array_shape<__NumD>::operator[] (__size_type index) 
const
{ return dims[index]; }


template <__size_type __NumD>
template <typename __T>
  inline
  __size_type 
  __multi_array_shape<__NumD>::check_and_cast(__T value )
  {
    if constexpr (std::is_signed_v<__T>) {
FINAL_PROJECT_ASSERT_MSG((value >= 0), "Negative value provided for an unsigned type.");
    }
return static_cast<__size_type>(value);
  }




} // end of namespace __detail


} // end of namespace multi_array
} // end of final_project



#endif // end of define FINAL_PROJECT_MULTIARRAY_TYPES_HPP_LIYIHAI