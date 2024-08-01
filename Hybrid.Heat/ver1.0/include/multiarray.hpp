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


/// @brief Give the prototypes of objects and functions.
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
    void Gather(multi_array::array_base<T, NumD> &, array_Cart<T, NumD> &, const Integer=0);


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
    using value_type  = T;
    using array       = __detail::__array<T, NumD>;
    using array_shape = __detail::__multi_array_shape<NumD>;

    private:
    std::unique_ptr<array> body;

    public:
    template <typename ... Exts>
    array_base( Exts ... );
    array_base( array_shape & );

    template <typename ... Args>
    T& operator()(Args ... args) { return (*body)(args...); }

    public:
    array&      data()                        { return *body; }
    array&      data()                  const { return *body; }
    size_type&  shape(size_type index)  const { return body->__shape[index]; }

    void saveToBinary(const String &)   const;

    // friend final_project::pde::Heat<T, NumD>;
    // friend final_project::pde::Naiver_Stokes<T, NumD>;

    friend mpi::array_Cart<T, NumD>;

  }; // class array_base
} // namespace multi_array



namespace mpi {


template <class T, size_type NumD>
  class array_Cart {
    public:
    typedef T                                       value_type;
    typedef topology::Cartesian<T, NumD>            topology_Cart;

    typedef mpi::__detail::__array_Cart<T, NumD>                loc_array;    // mpi details

    typedef multi_array::array_base<T, NumD>                    super_array;  
    typedef multi_array::__detail::__multi_array_shape<NumD>    array_shape;  // multi_array details

    private:
    std::unique_ptr<loc_array> body;

    public:
    array_Cart(environment &, array_shape &);
    
    template <typename ... Args>
    array_Cart(environment &, Args ...);

    template <typename ... Args>
    T& operator()(Args ... args) { return (*body)(args...); }

    public:
    void swap(array_Cart &);
    void swap(super_array &);

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
#include <cstring>
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


template <typename T, size_type NumD>
  void Gather(
    multi_array::array_base<T, NumD> & gather, 
    array_Cart<T, NumD> & loc,
    const Integer root)
  {

    MPI_Datatype buf_block, value_type {get_mpi_type<T>()};
    const Integer num_procs {loc.topology().num_procs}, dimension {loc.topology().dimension};

    Integer pid, i, j, k, dim, back {0};
    Integer s_list[NumD][num_procs], n_list[NumD][num_procs];
    std::array<Integer, NumD> Ns, starts_cpy, array_sizes, array_subsizes, array_starts, indexes;

// Gather basic information of sending entities

    if (num_procs == 1) { ++back; } // For handling the special case that there is only 1 process.
    array_starts.fill(0), indexes.fill(1);

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

// ------------------- Local Proc Sending to ROOT Proc ------------------- /
    if (loc.topology().rank != root)
    {

// - -  - - - - begin of create sending buffer datatypes - - - - - - - - //
for (dim = 0; dim < dimension; ++dim)
{
  array_sizes[dim] = loc.topology().__local_shape[dim];
  array_subsizes[dim] = Ns[dim];
}

MPI_Type_create_subarray( dimension, 
                          array_sizes.data(), 
                          array_subsizes.data(), 
                          array_starts.data(),
                          MPI_ORDER_C, value_type, &buf_block);
MPI_Type_commit(&buf_block);
// - -  - - - - end of create sending buffer datatypes - - - - - - - - //

if (NumD == 2) {
  MPI_Send( &loc(indexes[0], indexes[1]), 1, 
            buf_block, root, loc.topology().rank, loc.topology().comm_cart);
} 
  else if (NumD == 3)
{
  MPI_Send( &loc(indexes[0], indexes[1], indexes[2]), 1, 
            buf_block, root, loc.topology().rank, loc.topology().comm_cart);
}

MPI_Type_free(&buf_block);
    }


// ============= ROOT Gathering Information other Processes ============= //
    if (loc.topology().rank == root)
    {
      for ( pid = 0; pid < num_procs; ++pid)
      {
// ------------------- At the ROOT process ------------------- // 
        if (pid == root) // Local Memory Copy
        {
          for ( i=starts_cpy[0]; i <= loc.topology().ends[0] + back; ++i)
          {
if (NumD == 2)
{
  memcpy( &gather(i, starts_cpy[1]) , 
          &loc(i, starts_cpy[1])    ,   n_list[1][pid]*sizeof(T));
} 
  else if (NumD == 3)
{
  for ( j=starts_cpy[1]; j <= loc.topology().ends[1]+back; ++j)
    memcpy( &gather(i,j,starts_cpy[2]), 
            &loc(i,j,starts_cpy[2])   , n_list[2][pid]*sizeof(T));
}
          }
        } 
// ------------------- At the Other processes ------------------- // 
          else          // Recv From others
        {

// - -  - - - - begin of create receiving buffer datatypes - - - - - - - - //
for (dim = 0; dim < dimension; ++dim)
{
  array_sizes[dim] = loc.topology().__global_shape[dim];
  array_subsizes[dim] = n_list[dim][pid];
}

MPI_Type_create_subarray( dimension, 
                          array_sizes.data(), 
                          array_subsizes.data(), 
                          array_starts.data(),
                          MPI_ORDER_C, value_type, &buf_block);
MPI_Type_commit(&buf_block);
// - -  - - - - end of create receiving buffer datatypes - - - - - - - - //

if (NumD==2)
{
  MPI_Recv( &gather(s_list[0][pid], s_list[1][pid]),
            1, buf_block, pid, pid, loc.topology().comm_cart, MPI_STATUS_IGNORE);
} else if (NumD==3)
{
  MPI_Recv( &gather(s_list[0][pid], s_list[1][pid], s_list[2][pid]),
            1, buf_block, pid, pid, loc.topology().comm_cart, MPI_STATUS_IGNORE);
}

MPI_Type_free(&buf_block);
        }               // end of Recv From others
        
      }
    
    }

  }


} // namespace mpi
} // namespace final_project




#endif // end of define FINAL_PROJECT_MULTI_ARRAY_HPP_LIYIHAI





