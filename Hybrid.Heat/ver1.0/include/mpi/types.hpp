/// @file mpi/types.hpp
/// @brief Gives the Predefined MPI_Datatypes and Portable MPI_Datatypes
///     for communicating.
/// @date Jul. 27. 2024
/// @author LI Yihai
/// @see types.hpp
#ifndef FINAL_PROJECT_MPI_TYPES_HPP_LIYIHAI
#define FINAL_PROJECT_MPI_TYPES_HPP_LIYIHAI

#pragma once
#include <mpi.h>          // OpenMPI Library
#include <types.hpp>   // Final Project Header Files

namespace final_project {
namespace mpi {

///
/// @brief Predefined MPI Datatypes, 
///   INT, FLOAT, DOUBLE, BYTE, 
///   UNSIGNED SHORT, 
///   UNSIGNED UNSIGNED LONG LONG.
/// @tparam T Defined Datatypes for this Project
/// @return Return MPI_Datatype
/// 
template <typename T>
MPI_Datatype get_mpi_type(); 

/// @return MPI_INT
template <>
  inline 
  MPI_Datatype 
  get_mpi_type<Integer>()
{ return MPI_INT; }

/// @return MPI_FLOAT
template <>
  inline 
  MPI_Datatype
  get_mpi_type<Float>()
{ return MPI_FLOAT; }


/// @return MPI_DOUBLE
template <>
  inline 
  MPI_Datatype
  get_mpi_type<Double>()
{ return MPI_DOUBLE; }

/// @return MPI_BYTE
template <>
  inline 
  MPI_Datatype
  get_mpi_type<Byte>() 
{ return MPI_BYTE; }

/// @return MPI_UNSIGNED_SHORT
template <> 
  inline 
  MPI_Datatype
  get_mpi_type<Word>()
{ return MPI_UNSIGNED_SHORT; }

/// @return MPI_UNSIGNED
template <>
  inline 
  MPI_Datatype
  get_mpi_type<Dworld>()
{ return MPI_UNSIGNED; }

/// @return MPI_UNSIGNED_LONG_LONG
template <>
  inline 
  MPI_Datatype 
  get_mpi_type<Qworld>()
{ return MPI_UNSIGNED_LONG_LONG; }


} // end of namespace mpi
} // end of namespace final_project




#endif // end of define FINAL_PROJECT_MPI_TYPES_HPP_LIYIHAI