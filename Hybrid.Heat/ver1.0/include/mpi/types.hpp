

#ifndef FINAL_PROJECT_MPI_TYPES_HPP_LIYIHAI
#define FINAL_PROJECT_MPI_TYPES_HPP_LIYIHAI

#pragma once
#include <mpi.h>
#include "../types.hpp"

namespace final_project {
namespace mpi {


template <typename T>
MPI_Datatype get_mpi_type(); 


template <>
  inline 
  MPI_Datatype 
  get_mpi_type<final_project::Integer>()
{ return MPI_INT; }

template <>
  inline 
  MPI_Datatype
  get_mpi_type<final_project::Float>()
{ return MPI_FLOAT; }

template <>
  inline 
  MPI_Datatype
  get_mpi_type<final_project::Double>()
{ return MPI_DOUBLE; }

template <>
  inline 
  MPI_Datatype
  get_mpi_type<final_project::Byte>() 
{ return MPI_BYTE; }

template <> 
  inline 
  MPI_Datatype
  get_mpi_type<final_project::Word>()
{ return MPI_UNSIGNED_SHORT; }

template <>
  inline 
  MPI_Datatype
  get_mpi_type<final_project::Dworld>()
{ return MPI_UNSIGNED; }

template <>
  inline 
  MPI_Datatype 
  get_mpi_type<final_project::Qworld>()
{ return MPI_UNSIGNED_LONG_LONG; }

} // end of namespace mpi
} // end of namespace final_project










#endif // end of define FINAL_PROJECT_MPI_TYPES_HPP_LIYIHAI