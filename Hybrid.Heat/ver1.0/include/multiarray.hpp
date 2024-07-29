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

namespace final_project { namespace multi_array {


template <class T, size_type NumD>
  class array_base;

template <class T, size_type NumD>
  class array_Cart;

}}


namespace final_project { namespace multi_array {

/// @brief A base multi-dimension array class, interacted with USERs.
/// @tparam T The value type of array
/// @tparam NumD Number of dimensions.
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
} // namespace final_project


/// --------------------------------------------------
///
/// Definition of inline member functions
///

#include <fstream>
#include <assert.hpp>


namespace final_project { namespace multi_array {
  
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
} // namespace final_project




#endif // end of define FINAL_PROJECT_MULTI_ARRAY_HPP_LIYIHAI





