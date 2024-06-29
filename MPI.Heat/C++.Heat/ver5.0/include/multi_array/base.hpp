///
/// @file base.hpp
/// @brief This file define a skeleton of X ( > 0) Dimension array with basic operators.
/// 
///
/// @author LI Yihai
/// @version 5.0
/// @date Jun 26, 2024
///


#ifndef FINAL_PROJECT_MULTI_ARRAY_HPP_LIYIHAI
#define FINAL_PROJECT_MULTI_ARRAY_HPP_LIYIHAI


// ------------------------------- Header File ------------------------------- // 

#include "types.hpp"

namespace final_project {
namespace __detail {
namespace __multi_array {


typedef __types::__size_type __size_type;


/// @brief 
/// @tparam __T 
/// @tparam __NumD 
template <class __T, __size_type __NumD>
  class __array {
    public:
    typedef __T         value_type;
    typedef const __T   const_value_type;
    typedef __T&        reference;
    typedef const __T&  const_reference;
    typedef __T*        iterator;
    typedef const __T*  const_iterator;

    typedef __types::__multi_array_shape<__NumD> __super_array_shape;

    public:
    __super_array_shape     __shape;
    std::unique_ptr<__T[]> __data;

    public:
    /// @brief Default constructor of __array
    __array( );
    __array(__super_array_shape __shape);

    /// @brief Destructor of __array
    ~__array() = default;

    public:
    // Sizes
    __size_type size()        { return __shape.num_size(); }
    __size_type size() const  { return __shape.num_size(); }

    // Iterators
    iterator          begin()       { return __data.get(); }
    const_iterator    begin() const { return __data.get(); }
    const_iterator   cbegin() const { return __data.get(); }

    iterator            end()       { return __data.get() + this->size(); }
    const_iterator      end() const { return __data.get() + this->size(); }
    const_iterator     cend() const { return __data.get() + this->size(); }

    // Operators ()
    template <typename... Args>
    reference operator()(Args... args);
    reference operator[](__size_type index);

  public:
    void fill (const_reference value);   
    void assign (const_reference value); 

  public:
  template <class __U, __size_type __Dims>
  friend std::ostream& operator<<(std::ostream& os, const __array<__U, __Dims>& in);


  }; // class _array<_T, _NumD>


} // namespace __multi_array
} // namespace __detail
} // namespace final_project


// ------------------------------- Source File ------------------------------- // 

#include <iomanip>

namespace final_project {
namespace __detail {
namespace __multi_array {

template <class __T, __size_type __NumD>
  inline
  __array<__T, __NumD>::__array( )
  : __shape(), __data(nullptr) 
  { }


template <class __T, __size_type __NumD>
  inline
  __array<__T, __NumD>::__array(__super_array_shape __shape)
    : __shape {__shape}, __data {std::make_unique<__T[]>(__shape.num_size())} 
  { 
    // std::cout << "Constructing Array with number of elements " << __shape.num_size() << "\n";
  }


template <class __T, __size_type __NumD>
template <typename ... Args>
  inline
  __T& __array<__T, __NumD>::operator()(Args ... args) {
FINAL_PROJECT_ASSERT_MSG((sizeof...(args) == __NumD), "Number of arguments must match the number of dimensions.");

    __size_type indices[] = { static_cast<__size_type>(args)... };
    __size_type index = 0;
    __size_type multiplier = 1;

    for (__size_type i = 0; i < __NumD; ++i) 
    {
index += indices[__NumD - 1 - i] * multiplier;
multiplier *= __shape[__NumD - 1 - i];
    }

return __data[index];
  }

template <class __T, __size_type __NumD>
  inline 
  __T& __array<__T, __NumD>::operator[](__size_type index) {
FINAL_PROJECT_ASSERT_MSG((index < __shape.size()), "Index out of range.");
return __data[index]; 
  }

template <class __T, __size_type __NumD>
  inline 
  void __array<__T, __NumD>::fill(const __T& value) {
std::fill_n(begin(), size(), value);
  }

template <class __T, __size_type __NumD>
  inline 
  void __array<__T, __NumD>::assign(const __T& value)
  { fill(value); }


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


} // namespace _multi_array
} // namespace _detail
} // namespace final_project


#endif // define FINAL_PROJECT_MULTI_ARRAY_HPP_LIYIHAI