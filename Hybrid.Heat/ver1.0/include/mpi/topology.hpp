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



namespace final_project { namespace mpi {

namespace topology 
{
  typedef Dworld      size_type;
  
  template <typename T, UnsignedInteger NumD>
    struct Cartesian 
    {
      typedef multi_array::__detail::__multi_array_shape<NumD>    __array_shape;

      public:
      Integer dimension, num_procs;
      __array_shape __global_shape;

      MPI_Comm comm_cart;
      MPI_Datatype value_type {get_mpi_type<T>()};

      public: 
      __array_shape __local_shape;

      Integer rank;
      std::array<Integer, NumD> starts, ends, dims, periods, coordinates;
      std::array<Integer, NumD> nbr_src, nbr_dest;
      std::array<MPI_Datatype, NumD> halos;

      public:
      Cartesian();
      Cartesian(__array_shape &, environment &);

      bool operator==(const Cartesian &) const;
      bool operator!=(const Cartesian &) const;

      ~Cartesian();

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

/// @brief Helper Function, provides the decomposition routine.
auto Decomp = [](
  const Integer n, const Integer prob_size, const Integer rank, 
  Integer & s, Integer & e)
{
  Integer n_loc {n / prob_size}, deficit {n % prob_size};

  s = rank * n_loc + 1;
  s += ((rank < deficit) ? rank : deficit);

  if (rank < deficit) ++n_loc;
  e = s + n_loc - 1;

  if (e > n || rank == prob_size - 1) e = n;
  return 0;
};


  template <typename T, UnsignedInteger NumD>
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

  template <typename T, UnsignedInteger NumD>
    inline
    Cartesian<T, NumD>::~Cartesian()
    {
      UnsignedInteger i {0};
      // for (; i < dimension; ++i) {
      //   if (halos[i] != MPI_DATATYPE_NULL) 
      //     {MPI_Type_free(&halos[i]);}
      //   }

      // if (comm_cart != MPI_COMM_NULL) 
      // {
      //   MPI_Comm_free(&comm_cart);
      // }
    }


  template <typename T, UnsignedInteger NumD>
    inline 
    Cartesian<T, NumD>::Cartesian(
      __array_shape & global_shape, 
      environment & env)
  // : __global_shape(global_shape), __local_shape(global_shape)
    {
      UnsignedInteger i {0};
      dims.fill(0);
      
      MPI_Dims_create(env.size(), NumD, dims.data());
      MPI_Cart_create(env.comm(), NumD, dims.data(), periods.data(), 1, &comm_cart);
      MPI_Comm_size(comm_cart, &num_procs);

      MPI_Comm_rank(comm_cart, &rank);
      MPI_Cart_coords(comm_cart, rank, NumD, coordinates.data());

      std::array<Integer, NumD> array_size, array_sub_size, array_starts;


      ///
      std::cout << "Rank " << rank << "/" << num_procs 
                << " in " << dims[0] << " by " << dims[1]
                << " coordinate [" << coordinates[0] << ", " << coordinates[1] << "]"
                << " has neighbors in " 
                << std::endl;

    }


} // namespace topology

} // namespace topology
} // namespace final_project

#endif // end of define FINAL_PROJECT_MPI_TOPOLOGY_HPP_LIYIHAI