///
/// @file mpi/topology.hpp
/// @brief This file contains the details of types design, specifically 
/// for MPI Topology structure.
///
/// @author LI Yihai
/// @version 6.0 
/// @date Jul 28, 2024
///

#ifndef FINAL_PROJECT_MPI_TOPOLOGY_HPP_LIYIHAI
#define FINAL_PROJECT_MPI_TOPOLOGY_HPP_LIYIHAI

#pragma once
// OpenMPI Library
#include <mpi.h>
#include <array>

// Final Project Header Files
#include <types.hpp>
#include <multiarray/types.hpp>
#include <mpi/environment.hpp>
#include <mpi/types.hpp>


namespace final_project { namespace mpi {

namespace topology 
{

  /// @brief Cartesian struct contains the Basic Features of Cartesian @c MPI_Cart_Create
  ///         and also creates the basic routines of halo @c MPI_Datatype for communicating.
  ///         
  /// @tparam T The value type of topology structure members.
  /// @tparam NumD Number of dimensions.
  template <typename T, size_type NumD>
    struct Cartesian 
    {
      typedef multi_array::__detail::__multi_array_shape<NumD>    __array_shape;
      typedef std::array<Integer, NumD>                           __std_array_idx;
      typedef std::array<MPI_Datatype, NumD>                      __std_array_halos;

      // ------------------------- Global Entities ------------------------- //
      public:
      Integer dimension, num_procs;
      __array_shape __global_shape;

      MPI_Comm comm_cart;
      MPI_Datatype value_type {get_mpi_type<T>()};

      // ------------------------- Local Entities ------------------------- //
      public: 
      __array_shape __local_shape;

      __std_array_idx   starts, ends, dims = {0}, periods = {0};
      __std_array_idx   nbr_src, nbr_dest, coordinates;
      __std_array_halos halos;
      Integer rank;

      // -------------------------- Cons & Decons ------------------------- //
      public:
      Cartesian();
      Cartesian(const Cartesian &)              = delete;
      Cartesian(Cartesian &&)                   = delete;

      Cartesian& operator=(const Cartesian &)   = delete;
      Cartesian& operator=(Cartesian &&)        = delete; 

      explicit Cartesian(__array_shape &, environment &);
      ~Cartesian();

      // 
      // Bool operator==(const Cartesian &);
      // Bool operator!=(const Cartesian &);
    }; // struct Cartesian


} // namespace topology
} // namespace mpi
} // namespace final_project




///
/// Definitions of inline member functions
/// 
#include <assert.hpp>
#include <mpi/assert.hpp>

namespace final_project { namespace mpi {

namespace topology 
{

// template <typename T, size_type NumD>
//   inline Bool
//   Cartesian<T, NumD>::operator==(const Cartesian<T, NumD> & other)
// {
//   if (*this = &other) return true;

// }

/// @brief Default Constructor, fill with {0} value
/// @tparam T The value type
/// @tparam NumD Number of dimension {0}
template <typename T, size_type NumD>
  inline
  Cartesian<T, NumD>::Cartesian()
    : num_procs {0}, rank {0}, comm_cart {MPI_COMM_NULL}, 
      __local_shape(__global_shape), __global_shape()
  {
std::fill(      starts.begin(), starts.end(),       0);
std::fill(        ends.begin(), ends.end(),         0);
std::fill(        dims.begin(), dims.end(),         0);
std::fill(     periods.begin(), periods.end(),      0);
std::fill(     nbr_src.begin(), nbr_src.end(),      0);
std::fill(    nbr_dest.begin(), nbr_dest.end(),     0);
std::fill( coordinates.begin(), coordinates.end(),  0);
std::fill(halos.begin(),halos.end(), MPI_DATATYPE_NULL);
  }

/// @brief Destructor, Free halo @c MPI_Datatype and the @c MPI_Comm communicator
template <typename T, size_type NumD>
  inline
  Cartesian<T, NumD>::~Cartesian()
  {
    Integer i {0};
    for (; i < dimension; ++i) 
      if (halos[i] != MPI_DATATYPE_NULL) 
        {MPI_Type_free(&halos[i]);}

    if (comm_cart != MPI_COMM_NULL) 
      MPI_Comm_free(&comm_cart);
  }


/// @brief Constructor Cartesian Topology,
/// @tparam T 
/// @tparam NumD 
/// @param global_shape 
/// @param env The default mpi::environment
template <typename T, size_type NumD>
  inline 
  Cartesian<T, NumD>::Cartesian( __array_shape & global_shape, environment & env )
: __global_shape(global_shape), __local_shape(global_shape)
  {
    FINAL_PROJECT_ASSERT_MSG((NumD <= 4), "Number of dimension if out of range.");

/// @brief Helper Function, provides the decomposition routine.
auto Decomp = [](
  const Integer n, const Integer prob_size, const Integer rank, 
  Integer & s, Integer & e)
{
  Integer n_loc {n / prob_size}, remain {n % prob_size};

  s = rank * n_loc + 1;
  s += ((rank < remain) ? rank : remain);

  if (rank < remain) ++n_loc;
  e = s + n_loc - 1;

  if (e > n || rank == prob_size - 1) e = n;
  return 0;
};

    Integer i {0};
    dimension = static_cast<Integer>(NumD); // 0 <= Num < 4
    __std_array_idx array_size, array_sub_size, array_starts = {0};

    MPI_Dims_create(env.size(), dimension, dims.data());
    MPI_Cart_create(env.comm(), dimension, dims.data(), periods.data(), 1, &comm_cart);
    MPI_Comm_size(comm_cart, &num_procs);

    MPI_Comm_rank(comm_cart, &rank);
    MPI_Cart_coords(comm_cart, rank, dimension, coordinates.data());
    
    for (i = 0; i < dimension; ++i)
    {
      MPI_Cart_shift(comm_cart, i, 1, &nbr_src[i], &nbr_dest[i]);

      Decomp(__global_shape[i]-2, dims[i], coordinates[i], starts[i], ends[i]);
      __local_shape[i] = ends[i] - starts[i] + 1 + 2; // Include halos
      array_size[i] = __local_shape[i];
      array_sub_size[i] = array_size[i] - 2;
FINAL_PROJECT_MPI_ASSERT_GLOBAL((array_sub_size[i] > 0)); // sub size must bigger than 0
    }
    
    for (i = 0; i < dimension; ++i)
    {
      auto temp = array_sub_size[i];
      array_sub_size[i] = 1;
MPI_Type_create_subarray( dimension,  array_size.data(), 
                                      array_sub_size.data(), 
                                      array_starts.data(), 
                        MPI_ORDER_C, value_type, &halos[i]);

MPI_Type_commit(&halos[i]);
      array_sub_size[i] = temp;
    }
  }


} // namespace topology

} // namespace topology
} // namespace final_project

#endif // end of define FINAL_PROJECT_MPI_TOPOLOGY_HPP_LIYIHAI