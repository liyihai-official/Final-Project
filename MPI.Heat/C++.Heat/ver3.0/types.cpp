

#ifndef FINAL_PROJECT_TYPES_HPP_LIYIHAI
#define FINAL_PROJECT_TYPES_HPP_LIYIHAI

#include <mpi.h>
#include <memory>
#include <iostream>
#include "assert.cpp"

template<typename T>
MPI_Datatype get_mpi_type();

template<>
MPI_Datatype get_mpi_type<int>()    { return MPI_INT; }

template<>
MPI_Datatype get_mpi_type<float>()  { return MPI_FLOAT; }

template<>
MPI_Datatype get_mpi_type<double>() { return MPI_DOUBLE; }

namespace final_project
{
  namespace _detail 
  {

// General size types
    typedef std::size_t _size_type;


// Multi-dimension Array shape Types
template <_size_type NumDims>
struct _multi_array_shape 
{
  typedef final_project::_detail::_size_type _size_type;

  std::unique_ptr<_size_type[]> sizes;

  // _multi_array_shape() : sizes(std::make_unique<_size_type[]>(NumDims)) {}

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

//   _multi_array_shape& operator= (const _multi_array_shape& other)
//   {
//     if (this != &other)
//     {
// FINAL_PROJECT_ASSERT_MSG((NumDims == other.dim()), "Dimension mismatch in assignment. ");

// sizes = std::make_unique<_size_type[]>(NumDims);
// std::copy(other.sizes.get(), other.sizes.get()+ NumDims, sizes.get());
//     }

//     return *this;
//   }

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

// MPI_TOPOLOGY Types
template <typename T>
struct _mpi_topology
{
  int rank, num_proc, dimension;
  std::unique_ptr<int[]> neighbors, starts, ends, coordinates;

  MPI_Comm comm;
  MPI_Datatype type{get_mpi_type<T>()};
};




  }
}
#endif