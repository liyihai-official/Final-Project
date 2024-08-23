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

    typedef       __T         value_type;
    typedef const __T         const_value_type;
    typedef       __T&        reference;
    typedef const __T&        const_reference;
    typedef       __T*        iterator; 
    typedef const __T*        const_iterator;

    typedef __multi_array_shape<__NumD> __array_shape;

  public: // Member Variables
    __array_shape                 __shape;
    std::unique_ptr<value_type[]> __data;

    static constexpr __size_type  __dimension {__NumD};

  public: // Cons & Decons 
    __array();
    __array(__array_shape);
    __array(const __array &);
    __array& operator=(const __array&);
    ~__array() = default;


  public: // Operators
    template <typename ... Exts>
    reference operator()(Exts ...);
    reference operator[](__size_type);

    iterator          begin()       noexcept { return __data.get(); }
    const_iterator    begin() const noexcept { return __data.get(); }
    const_iterator   cbegin() const noexcept { return __data.get(); }

    iterator            end()       noexcept { return __data.get() + this->size(); }
    const_iterator      end() const noexcept { return __data.get() + this->size(); }
    const_iterator     cend() const noexcept { return __data.get() + this->size(); }

  public: // Member Functions
    __super_size_type size()  const noexcept { return __shape.size(); }
    Integer       get_flat_index( std::array<Integer, __NumD> & );

  public: // Member Functions for Features
    
    void swap(__array &)          noexcept;
    void fill(const_reference)    noexcept;
    void assign(const_reference)  noexcept;

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
  __array<__T, __NumD>::__array(__array_shape __in_shape)
: __shape {__in_shape},
  __data {std::make_unique<value_type[]>(__in_shape.size())}
  {}

template <class __T, __size_type __NumD>
  inline 
   __array<__T, __NumD>::__array(const __array & other)
: __shape(other.__shape),
  __data(std::make_unique<value_type[]>(other.size()))
{
  std::copy(other.__data.get(), other.__data.get() + other.size(), __data.get()); // Copy data
}

// Operators
template <class __T, __size_type __NumD>
template <typename ... Exts>
  inline
  __T& 
  __array<__T, __NumD>::operator()(Exts ... exts)
{
  FINAL_PROJECT_ASSERT_MSG(
    (sizeof...(exts) == __NumD),
    "Number of Arguments must Match the dimension."
  );

  __size_type index {0}, i {0};
  __size_type multiplier {1};
  __size_type indices[] = { __shape.check_and_cast(exts) ... };

  for (; i < __NumD; ++i)
  {
index += indices[__NumD - 1 - i] * multiplier;
multiplier *= __shape[__NumD - 1 - i];
  }

  return __data[index];
}

template <class __T, __size_type __NumD>
  inline 
  Integer
  __array<__T, __NumD>::get_flat_index(std::array<Integer, __NumD> & indexes)
{
  Integer index {0}, i {0};
  Integer multiplier {1};

  for (; i < __NumD; ++i)
  {
FINAL_PROJECT_ASSERT_MSG((indexes[__NumD - 1 - i] >= 0), "Indexing must be none-negative number.\n");
    index += indexes[__NumD - 1 - i] * multiplier;
    multiplier *= __shape[__NumD - 1 - i];
  }
  return index;
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
  inline __array<__T, __NumD>& 
  __array<__T, __NumD>::operator=(const __array& other)
{
  if (this != &other)
  {
__shape = other.__shape;
__data = std::make_unique<value_type[]>(other.size());
std::copy(other.__data.get(), other.__data.get()+other.size(), __data.get());
  }
  return *this;
}

template <class __T, __size_type __NumD>
  inline 
  void __array<__T, __NumD>::fill(const_reference value) noexcept
{ std::fill_n(begin(), size(), value); }

template <class __T, __size_type __NumD>
  inline 
  void __array<__T, __NumD>::assign(const_reference value) noexcept
{ fill(value); }

template <class __T, __size_type __NumD>
  inline
  void __array<__T, __NumD>::swap(__array & other) noexcept
{
  __data.swap(other.__data);
  __shape.swap(other.__shape);
}


/// @brief Recursively print multi-array
/// @tparam __U Value types 
/// @tparam __Dims Dimension types
/// @param os std::ostream
/// @param in Input multi_array
template <class __U, __size_type __Dims>
std::ostream& operator<<(std::ostream& os, const __array<__U, __Dims>& in) 
{
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
  os << std::fixed << std::setprecision(5) << std::setw(12) << (arr.__data[offset + i]);
}
os << " |\n";
        } else {
for (__size_type i = 0; i < arr.__shape[current_dim]; ++i) 
{
  self(self, arr, current_dim + 1, offset + i * arr.__shape.strides[current_dim]);
}
os << "\n";
    }
  }; // End of Helper Function

  // Start recursive printing
  print_recursive(print_recursive, in, 0, 0);

  return os;
}



} // end of namespace __detail



} // end of namespace multi_array
} // end of final_project





#endif // end of define FINAL_PROJECT_MULTIARRAY_BASE_HPP_LIYIHAI