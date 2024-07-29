///
/// @file multiarray/base.hpp
///
///



#ifndef FINAL_PROJECT_MULTIARRAY_BASE_HPP_LIYIHAI
#define FINAL_PROJECT_MULTIARRAY_BASE_HPP_LIYIHAI

#pragma once
// Standard Libraries
#include <memory>

// Final Project Header Files
#include <types.hpp>
#include <multiarray/types.hpp>
#include <assert.hpp>


namespace final_project { namespace multi_array {


namespace __detail {


/// @brief The routines and features of basic multi-array
/// @tparam __T The Datatype of values.
/// @tparam __NumD The number of dimensions.
template <class __T, __size_type __NumD>
  class __array {

  public: // Member Datatypes
    typedef __T         value_type;
    typedef const __T   const_value_type;
    typedef __T&        reference;
    typedef const __T&  const_reference;
    typedef __T*        iterator;
    typedef const __T*  const_iterator;

    typedef __multi_array_shape<__NumD> __array_shape;

  public: // Member Variables
    __array_shape               __shape;
    std::unique_ptr<value_type[]> __data;

  public: // Cons & Decons 
    __array();
    __array(__array_shape &);

    ~__array() = default;


  public: // Operators
    template <typename ... Args>
    reference operator()(Args ...);
    reference operator[](__size_type);
  
    iterator          begin()       { return __data.get(); }
    const_iterator    begin() const { return __data.get(); }
    const_iterator   cbegin() const { return __data.get(); }

    iterator            end()       { return __data.get() + this->size(); }
    const_iterator      end() const { return __data.get() + this->size(); }
    const_iterator     cend() const { return __data.get() + this->size(); }

  public: // Member Functions
    __super_size_type size() const { return __shape.size(); }

  public: // Member Functions for Features
    void swap(__array &);
    void fill(const_reference);
    void assign(const_reference);


  template <class __U, __size_type __Dims>
  friend std::ostream & operator<<(std::ostream &, const __array<__U, __Dims> &);
  }; // end if class __array

} // end of namespace __detail



} // end of namespace multi_array
} // end of final_project





/// --------------------------------------------------
///
/// Definition of inline member functions
///

#include <iomanip>

namespace final_project { namespace multi_array {

namespace __detail {


template <class __T, __size_type __NumD>
  inline
  __array<__T, __NumD>::__array()
  : __shape(), __data(nullptr) 
  {}

template <class __T, __size_type __NumD>
  inline
  __array<__T, __NumD>::__array(__array_shape & __in_shape)
  : __shape {__in_shape},
    __data {std::make_unique<value_type[]>(__in_shape.size())}
  {}


// Operators
template <class __T, __size_type __NumD>
template <typename ... Args>
  inline
  __T& 
  __array<__T, __NumD>::operator()(Args ... args)
  {
    FINAL_PROJECT_ASSERT_MSG(
      (sizeof...(args) == __NumD),
      "Number of Arguments must Match the dimension."
    );

    __size_type index {0}, i {0};
    __size_type multiplier {1};
    __size_type indices[] = { __shape.check_and_cast(args) ... };

    for (; i < __NumD; ++i)
    {
index += indices[__NumD - 1 - i] * multiplier;
multiplier *= __shape[__NumD - 1 - i];
    }

    return __data[index];
  }


template <class __T, __size_type __NumD>
  inline 
  __T& __array<__T, __NumD>::operator[](__size_type index) 
  {
    FINAL_PROJECT_ASSERT_MSG(
      (index < __shape.size()), 
      "Index out of range."
    );
    return __data[index];   
  }


template <class __T, __size_type __NumD>
  inline 
  void __array<__T, __NumD>::fill(const_reference value) 
{ std::fill_n(begin(), size(), value); }

template <class __T, __size_type __NumD>
  inline 
  void __array<__T, __NumD>::assign(const_reference value)
{ fill(value); }


/// @brief Recursively print multi-array
/// @tparam __U Value types 
/// @tparam __Dims Dimension types
/// @param os std::ostream
/// @param in Input multi_array
template <class __U, __size_type __Dims>
std::ostream& operator<<(std::ostream& os, const __array<__U, __Dims>& in) {

  // Helper Function to print multi-dimensional array
  auto print_recursive = [&](
    auto&& self, const __array<__U, __Dims>& arr, 
    __size_type current_dim, 
    __size_type offset) -> void 
  {
if (current_dim == __Dims - 1) 
{
  os << "|";
  for (__size_type i = 0; i < arr.__shape[current_dim]; ++i) 
{
os << std::fixed << std::setprecision(5) << std::setw(9) << (arr.__data[offset + i]);
  }
  os << " |\n";
} 
else 
{
  for (__size_type i = 0; i < arr.__shape[current_dim]; ++i) 
  {
    __size_type next_offset = offset;
    for (__size_type j = current_dim + 1; j < __Dims; ++j) {
      next_offset *= arr.__shape[j];
    }
    next_offset += i * arr.__shape[current_dim + 1];

    self(self, arr, current_dim + 1, next_offset);
  }
  os << "\n";
}
  }; // End of Helper Function


  print_recursive(print_recursive, in, 0, 0);  // Print Recursively

  return os;
}




} // end of namespace __detail



} // end of namespace multi_array
} // end of final_project





#endif // end of define FINAL_PROJECT_MULTIARRAY_BASE_HPP_LIYIHAI