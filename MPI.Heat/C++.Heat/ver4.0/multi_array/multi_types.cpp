#ifndef FINAL_PROJECT_MULTI_TYPES_HPP_LIYIHAI
#define FINAL_PROJECT_MULTI_TYPES_HPP_LIYIHAI

#include <memory>

#include <iostream> //

#include "assert"
#include "types"

namespace final_project
{
  namespace _detail 
  {

// Multi-dimension Array shape Types
template <_size_type NumDims>
struct _multi_array_shape 
{
  typedef final_project::_detail::_size_type _size_type;

  std::unique_ptr<_size_type[]> sizes;

  template <typename ... Args>
  _multi_array_shape(Args ... args) 
    : sizes(std::make_unique<_size_type[]>(NumDims))
  {

FINAL_PROJECT_ASSERT_MSG((sizeof...(args) == NumDims), "Number of arguments must match the number of dimensions.");

    _size_type temp[] = { static_cast<_size_type>(args)... };
    std::copy(temp, temp + NumDims, sizes.get());
    
  }

  _multi_array_shape(const _multi_array_shape& other)
    : sizes (std::make_unique<_size_type[]>(NumDims))
  {
std::copy(other.sizes.get(), other.sizes.get() + NumDims, sizes.get());
  }

  _size_type& operator[] (_size_type index) { return sizes[index]; }

  const _size_type& operator[] (_size_type index) const { return sizes[index]; }

  const _size_type dim() const { return NumDims; }

  const _size_type size() const {
    _size_type total {1} ;
    for (_size_type i = 0; i < NumDims; ++i) total *= sizes[i];
    return total;
  }

  bool is_in(_size_type index) const { return index < this->size(); }  
  
};


  }
}
#endif