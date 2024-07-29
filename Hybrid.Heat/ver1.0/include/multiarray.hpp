///
///
/// @file multiarray.hpp
/// @brief The header file provides object @class of multi dimension array
///        for Users. Also, the routine of the hybrid @class of multi dimension 
///        and @c MPI_Cart MPI topology struture.
/// @author LI Yihai
///
///

#ifndef FINAL_PROJECT_MULTI_ARRAY_HPP_LIYIHAI
#define FINAL_PROJECT_MULTI_ARRAY_HPP_LIYIHAI

#pragma once

// Final Project Header Files
#include <types.hpp>
#include <multiarray/base.hpp>
#include <multiarray/types.hpp>

namespace final_project { 
  
  
namespace multi_array {
  /// @brief A base multi-dimension array class, interacted with USERs.
  /// @tparam T The value type of array.
  /// @tparam NumD Number of dimensions.
  template <class T, size_type NumD>
    class array_base;
} // namespace multi_array

namespace mpi {

  /// @brief An array routine based on @c MPI_Cart Cartesian topology struture.
  /// @tparam T The value type of array.
  /// @tparam NumD The number of dimensions.
  template <class T, size_type NumD>
    class array_Cart;


  /// @brief Gather the distributed arrays on every processes in @c MPI_Comm environment.
  /// @tparam T the value type of arrays.
  /// @tparam NumD Number of dimension
  /// @param  gather The collective array.
  /// @param  cart  The distributed array on every processes.
  template <typename T, size_type NumD>
    void Gather(array_base<T, NumD> &, array_Cart<T, NumD> &);

} // namespace mpi


} // namespace final_project


// Final Project Header Files
#include <mpi/types.hpp>
#include <mpi/topology.hpp>
#include <mpi/multiarray.hpp>
#include <mpi/environment.hpp>

namespace final_project { 
  
  
namespace multi_array {
template <class T, size_type NumD>
  class array_base 
  {
    // friend final_project::PDE::Heat<T, NumD>;
    // friend final_project::PDE::Naiver_Stokes<T, NumD>;

    private:
    typedef T                                     value_type;
    typedef __detail::__array<T, NumD>            array;
    typedef __detail::__multi_array_shape<NumD>   array_shape;

    private:
    std::unique_ptr<array> body;

    public:
    template <typename ... Args>
    array_base( Args ... );
    array_base( array_shape & );

    public:
    array& data()                       { return *body; }
    array& data()                 const { return *body; }
    size_type&  shape(size_type index)  const { return body->__shape[index]; }

    void saveToBinary(const String &) const;
  }; // class array_base
} // namespace multi_array



namespace mpi {

template <class T, size_type NumD>
  class array_Cart {

    // friend final_project::PDE::Heat<T, NumD>;
    // friend final_project::PDE::Naiver_Stokes<T, NumD>;

    public:
    typedef T                                       value_type;
    typedef topology::Cartesian<T, NumD>            topology_Cart;

    typedef mpi::__detail::__array_Cart<T, NumD>                loc_array;    // mpi details
    typedef multi_array::__detail::__multi_array_shape<NumD>    array_shape;  // multi_array details

    private:
    std::unique_ptr<loc_array> body;

    public:
    array_Cart(environment &, array_shape &);
    
    template <typename ... Args>
    array_Cart(environment &, Args ...);

    public:
    void swap(array_Cart &);

    array_Cart& array();
    array_Cart& array() const;

    topology_Cart& topology() const;
  }; // class array_Cart

} // namespace mpi









} // namespace final_project


/// --------------------------------------------------
///
/// Definition of inline member functions
///

#include <fstream>
#include <assert.hpp>


namespace final_project { 
  
  
  
namespace multi_array {
  
template <class T, size_type NumD>
  inline
  array_base<T, NumD>::array_base(array_shape & shape)
: body (std::make_unique<array>(shape))
  { FINAL_PROJECT_ASSERT_MSG((NumD < 4), "Invalid Dimension of Array."); }

template <class T, size_type NumD>
template <typename ... Args>
  inline 
  array_base<T, NumD>::array_base(Args ... args)
: body(std::make_unique<array>(array_shape(args ...)))
  { FINAL_PROJECT_ASSERT_MSG((NumD < 4), "Invalid Dimension of Array."); }

/// @brief Save the array to given file in binary mode
/// @tparam T The value type
/// @tparam NumD The number of Dimensions
/// @param filename 
template <class T, size_type NumD>
  inline
  void 
  array_base<T, NumD>::saveToBinary(const String & filename) 
  const 
{
  std::ostream ofs (filename, std::ios::binary);
  FINAL_PROJECT_ASSERT(ofs);

  for (size_type i = 0 ; i < NumD; ++i)
  {
    auto temp {body->__shape[i]};
    ofs.write(reinterpret_cast<const Char*>(&temp), sizeof((temp)));
  }
  
  ofs.write(reinterpret_cast<const Char*>(
    body->begin()), body->size() * sizeof(T)
  );
}


} // namespace multi_array



namespace mpi {

template <class T, size_type NumD>
  inline
  array_Cart<T, NumD>::array_Cart(environment & env, array_shape & glob_shape)
: body(std::make_unique<array_Cart>(glob_shape, env))
{ FINAL_PROJECT_ASSERT((NumD < 4)); }


template <class T, size_type NumD>
template <typename ... Args>
  inline
  array_Cart<T, NumD>::array_Cart(environment & env, Args ... args)
: body(
  [&](){
    array_shape shape(args...);
    return std::make_unique<array_Cart>(shape, env);
  }())
{ FINAL_PROJECT_ASSERT((NumD < 4)); }

  
} // namespace mpi




} // namespace final_project




#endif // end of define FINAL_PROJECT_MULTI_ARRAY_HPP_LIYIHAI





