///
/// @file mpi/multiarray.hpp
/// @brief 
///
/// @author LI Yihai
///


#ifndef FINAL_PROJECT_MPI_MULTIARRAY_HPP_LIYIHAI
#define FINAL_PROJECT_MPI_MULTIARRAY_HPP_LIYIHAI



#pragma once
// Final Project Header Files
#include <types.hpp>

#include <mpi/types.hpp>
#include <mpi/topology.hpp>

#include <multiarray/base.hpp>
#include <multiarray/types.hpp>

namespace final_project { namespace mpi {


namespace __detail {

/// @brief The implementation of multi-dimension array, based on 
///         @c MPI_Cart topology structure.
/// @tparam __T The value type.
/// @tparam __NumD The number of dimensions.
template <class __T, __size_type __NumD>
  class __array_Cart {
    public:
    typedef multi_array::__detail::__array<__T, __NumD> __array;
    typedef multi_array::__detail::__multi_array_shape<__NumD> __shape;
    typedef topology::Cartesian<__T, __NumD>            __Cart;

    public:
    __Cart  __loc_Cart;
    __array __loc_array;

    public:
    __array_Cart(__shape &, environment &);

    // Member Functions
    public:
    void swap(__array_Cart &);

    public: // friend Functions
    template <class __U, __size_type __Dims>
    friend std::ostream& operator<<(std::ostream &, const __array_Cart<__U, __Dims> &);

  }; // class __array_Cart


} // end of namespace __detail
} // end of namespace mpi
} // end of final_project




/// --------------------------------------------------
///
/// Definition of inline member functions
///
///

// Unix Standard Libraries
#include <unistd.h>

// Final Project Header Files
#include <assert.hpp>
#include <mpi/assert.hpp>



namespace final_project { namespace mpi {
namespace __detail {


template <class __T, __size_type __NumD>
  inline
  __array_Cart<__T, __NumD>::__array_Cart(__shape & __glob_shape, environment & env)
: __loc_Cart(__glob_shape, env), 
  __loc_array(__loc_Cart.__local_shape) {}

template <class __T, __size_type __NumD>
  inline
  void __array_Cart<__T, __NumD>::swap(__array_Cart & other)
  {
    FINAL_PROJECT_ASSERT_MSG(
      (__loc_Cart == other.__loc_Cart),
      "Match MPI Topology Structure required for swapping array_Cart."
    );
    __loc_array.swap(other.__loc_array);
  }



template <class __U, __size_type __Dims>
  std::ostream& operator<<(std::ostream& os, const __array_Cart<__U, __Dims>& in)
  {
    MPI_Barrier(in.__loc_Cart.comm_cart);
    sleep(1);
  os << "Attempting to print array in order \n";
  MPI_Barrier(in.__loc_Cart.comm_cart);

  for (int i = 0; i < in.__loc_Cart.num_procs; ++i)
  {
    if ( i == in.__loc_Cart.rank )
    {
  os
  << "\nPROC : " << in.__loc_Cart.rank << " of " 
  << in.__loc_Cart.num_procs <<  " is Printing \n" 
  << in.__loc_array;
    }
    fflush(stdout);
    sleep(0.1);
    MPI_Barrier(in.__loc_Cart.comm_cart);
  }

  return os;
  }




} // end of namespace __detail
} // end of namespace mpi
} // end of final_project




#endif // end of define FINAL_PROJECT_MPI_MULTIARRAY_HPP_LIYIHAI