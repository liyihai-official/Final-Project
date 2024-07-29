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
    array& data()                             { return *body; }
    array& data()                       const { return *body; }
    size_type&  shape(size_type index)  const { return body->__shape[index]; }

    void saveToBinary(const String &)   const;

    // friend final_project::pde::Heat<T, NumD>;
    // friend final_project::pde::Naiver_Stokes<T, NumD>;

  }; // class array_base
} // namespace multi_array



namespace mpi {

template <class T, size_type NumD>
  class array_Cart {
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

    loc_array& array()              { return *body; }
    loc_array& array()        const { return *body; }

    topology_Cart& topology() const { return body->__loc_Cart; }

    // friend final_project::PDE::Heat<T, NumD>;
    // friend final_project::PDE::Naiver_Stokes<T, NumD>;

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
: body(std::make_unique<loc_array>(glob_shape, env))
{ FINAL_PROJECT_ASSERT((NumD < 4)); }


template <class T, size_type NumD>
template <typename ... Args>
  inline
  array_Cart<T, NumD>::array_Cart(environment & env, Args ... args)
: body(
  [&](){
    array_shape shape(args...);
    return std::make_unique<loc_array>(shape, env);
  }())
{ FINAL_PROJECT_ASSERT((NumD < 4)); }


/// @brief Gather the distributed arrays on every processes in @c MPI_Comm environment.
/// @tparam T the value type of arrays.
/// @tparam NumD Number of dimension
/// @param  gather The collective array.
/// @param  cart  The distributed array on every processes.
template <typename T, size_type NumD>
  void Gather(multi_array::array_base<T, NumD> &, array_Cart<T, NumD> &, const Integer=0);

template <typename T, size_type NumD>
  void Gather(
    multi_array::array_base<T, NumD> & gather, 
    array_Cart<T, NumD> & loc,
    const Integer root)
  {
    MPI_Datatype sbuf_block, value_type {get_mpi_type<T>()};

    // if (loc.topology().num_procs == 1) {
    //   *(gather.body).swap(*(loc.body).__loc_array);
    // }

    Integer pid, i, j, k, dim, num_proc {loc.topology().num_procs}, dimension {loc.topology().dimension};
    Integer s_list[NumD][num_proc], n_list[NumD][num_proc];
    std::array<Integer, NumD> Ns, starts_cpy, array_sizes, array_subsizes, array_starts = {0}, indexes = {1};

    for (dim = 0; dim < dimension; ++dim)
    {
      Ns[dim] = loc.topology().__local_shape[dim] - 2;
      starts_cpy[dim] = loc.topology().starts[dim];

      if (starts_cpy[dim] == 1)
      {
        -- starts_cpy[dim];
        -- indexes[dim];
        ++ Ns[dim];
      }

      if (loc.topology().ends[dim] == loc.topology().__global_shape[dim] - 2)
        ++ Ns[dim];

      MPI_Datatype gtype {get_mpi_type<Integer>()};
      MPI_Gather(&starts_cpy[dim], 1, gtype, s_list[dim], 1, gtype, root, loc.topology().comm_cart);
      MPI_Gather(&Ns[dim], 1, gtype, n_list[dim], 1, gtype, root, loc.topology().comm_cart);  
    }

    if (loc.topology().rank == root)
    {

    }

    if (loc.topology().rank != root)
    {
      for ( pid = 0; pid < num_proc; ++pid)
      {
        if (pid == root)
        {
          for ( i=starts_cpy[0]; i <= loc.topology().ends[0]; ++i)
          {
            if (NumD == 2)
            {
              // memcpy( )
            }
          }
        }
      }
    }

  }



} // namespace mpi
} // namespace final_project




#endif // end of define FINAL_PROJECT_MULTI_ARRAY_HPP_LIYIHAI





