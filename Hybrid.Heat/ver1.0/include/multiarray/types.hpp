///
/// @file   multiarray/types.hpp
/// @brief  This file defines datatypes for multi-array objects
/// @author LI Yihai
///

#ifndef FINAL_PROJECT_MULTIARRAY_TYPES_HPP_LIYIHAI
#define FINAL_PROJECT_MULTIARRAY_TYPES_HPP_LIYIHAI

#pragma once
// Standard Libraries
#include <vector>       
#include <types.hpp>    // Final Project Header Files

namespace final_project { namespace multi_array {

namespace __detail {
  typedef size_type         __size_type;
  typedef super_size_type   __super_size_type;

  template <__size_type __NumD>
    struct __multi_array_shape;
}
}} 


namespace final_project { namespace multi_array {
namespace __detail {

/// @brief A shape viewer of multi-array which is __NumD dimension(s).
/// @tparam __NumD Template __size_type number, the dimensions.
template <__size_type __NumD>
  struct __multi_array_shape
  {
    typedef const __size_type       const_size_type;
    typedef       __size_type&      reference;
    typedef const __size_type&      const_reference;

    // Member Variables
    std::vector<__size_type> dims, strides;

    // Cons & Decons
    __multi_array_shape()                                       noexcept; 
    __multi_array_shape(const __multi_array_shape&)             noexcept;
    __multi_array_shape(__multi_array_shape &&)                 noexcept;
    __multi_array_shape& operator=(const __multi_array_shape&)  noexcept;
    __multi_array_shape& operator=(__multi_array_shape &&)      noexcept;

    /// @brief Construct from given vector
    /// @param  dims A std::vector of dimensions.
    explicit __multi_array_shape(const std::vector<__size_type> &)       noexcept;

    /// @brief Construct dims from packed arguments.
    template <typename ... Exts>
    explicit __multi_array_shape( Exts ... );

    // Operators
    Bool operator==(const __multi_array_shape &)                noexcept;
    Bool operator!=(const __multi_array_shape &)                noexcept;

    // Member Access
    reference       operator[] (__size_type);
    const_reference operator[] (__size_type) const;

    // Capacity
    __super_size_type size() const; // Using Super for avoiding overflow

    // Modifiers 
    void swap(__multi_array_shape &)                            noexcept;

    /// @brief A helper Function for ensuring none-negative inputs
    template <typename __T>
    __size_type check_and_cast(__T);

    /// @brief Determine the Strides
    std::function<void()> compute_strides = [this]() 
    {
      this->strides.resize(this->dims.size());
      this->strides[this->dims.size() - 1] = 1;

      for (__size_type i = __NumD - 1; i > 0; --i)
        this->strides[i-1] = this->strides[i] * this->dims[i];
    };

  };
  



} // end of namespace __detail
} // end of namespace multi_array
} // end of final_project





/// --------------------------------------------------
///
/// Definition of inline member functions
///
#include <algorithm>
#include <assert.hpp>

namespace final_project { namespace multi_array {


namespace __detail { 

template <__size_type __NumD>
  inline
  __multi_array_shape<__NumD>::__multi_array_shape() noexcept
{ dims.resize(0); strides.resize(0); }

template <__size_type __NumD>
  inline
  __multi_array_shape<__NumD>::__multi_array_shape(
    const __multi_array_shape & other
  ) noexcept
{ 
  dims = other.dims;
  compute_strides();
}

template <__size_type __NumD>
  inline 
  __multi_array_shape<__NumD>::__multi_array_shape(
    __multi_array_shape && other
  )  noexcept
: dims(std::move(other.dims)), 
  strides(std::move(other.strides))
  {}


template <__size_type __NumD>
  inline __multi_array_shape<__NumD>&
  __multi_array_shape<__NumD>::operator=(
    const __multi_array_shape & other
  ) noexcept
{
  if (this != &other)
  {
dims.resize(other.dims.size());
strides.resize(other.strides.size());
std::copy(other.dims.cbegin(), other.dims.cend(), dims.begin());
std::copy(other.strides.cbegin(), other.strides.cend(), strides.begin());
  }
  return *this;
}


template <__size_type __NumD>
  inline __multi_array_shape<__NumD>&
  __multi_array_shape<__NumD>::operator=(
    __multi_array_shape && other
  )  noexcept
{
  if (this != &other)
  {
    dims = std::move(other.dims);
    strides = std::move(other.strides);

    other.dims.clear();
    other.strides.clear();
  } 
  return *this;
}



template <__size_type __NumD>
template <typename ... Exts>
  inline 
  __multi_array_shape<__NumD>::__multi_array_shape(Exts ... exts)
{ 
  FINAL_PROJECT_ASSERT_MSG(
    (sizeof...(exts) == __NumD),
    "Number of Arguments must Match the dimension."
  );  

  // cast the inputs (non-negative) into __size_type (uint64_t)
  dims = { check_and_cast(exts)... };
  compute_strides();
}

template <__size_type __NumD>
  inline 
  __multi_array_shape<__NumD>::__multi_array_shape(
    const std::vector<__size_type> & dimensions
  ) noexcept
{
  dims.resize(dimensions.size());
  std::copy(dimensions.cbegin(), dimensions.cend(), dims.begin());
  compute_strides();
}

template <__size_type __NumD>
  inline 
  __super_size_type
  __multi_array_shape<__NumD>::size()
  const
{ 
  __super_size_type total {1};
  for (auto & elem : dims) 
  { total *= static_cast<__super_size_type>(elem); }

  return total;
}

template <__size_type __NumD>
  inline void 
  __multi_array_shape<__NumD>::swap(
    __multi_array_shape & other
  )   noexcept
{
  dims.swap(other.dims);
  strides.swap(other.strides);
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
  inline 
  Bool 
  __multi_array_shape<__NumD>::operator==(
    const __multi_array_shape & other
  ) noexcept
{ return (other.dims == dims && other.strides == strides); }

template <__size_type __NumD>
  inline 
  Bool 
  __multi_array_shape<__NumD>::operator!=(
    const __multi_array_shape & other
  ) noexcept
{ return (other.dims != dims || other.strides != strides); }


template <__size_type __NumD>
template <typename __T>
  inline
  __size_type 
  __multi_array_shape<__NumD>::check_and_cast(__T value )
{
  if constexpr (std::is_signed_v<__T>) 
{ FINAL_PROJECT_ASSERT_MSG((value >= 0), "Negative value provided for an unsigned type."); }
  return static_cast<__size_type>(value);
}




} // end of namespace __detail
} // end of namespace multi_array
} // end of final_project



#endif // end of define FINAL_PROJECT_MULTIARRAY_TYPES_HPP_LIYIHAI