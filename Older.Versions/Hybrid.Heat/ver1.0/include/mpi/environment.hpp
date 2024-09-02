/// 
/// @file mpi/environment.hpp 
/// 
/// @brief This Header file provides the MPI @c environment class included the basic 
///       initializations and finalizations. Also, provides the routines of inline 
///       member function of class.
///

#ifndef FINAL_PROJECT_ENVRIONMENT_HPP_LIYIHAI
#define FINAL_PROJECT_ENVRIONMENT_HPP_LIYIHAI

#pragma once
#include <mpi.h>
#include <optional>
#include <types.hpp>
#include <assert.hpp>
#include <mpi/assert.hpp>

///
/// class definition @c environment .
///
namespace final_project { namespace mpi {

///
/// @brief Initialize and Finalize MPI environment.
///       This @c environment class is capsuled with @c MPI_Init and @c MPI_Finalize
///       Will be used in the main program and called only once.
class environment {
public:
environment() = delete;
environment(const environment &)  = delete;
environment(environment &&)       = delete;

/// @brief Initialize the MPI environment.
environment(
  Integer& argc, Char** &argv, 
  Integer host_prank = 0,
  Integer required = MPI_THREAD_SINGLE)
  {
Integer provided;
MPI_Init_thread(&argc, &argv, required, &provided);
FINAL_PROJECT_MPI_INIT_CHECK();

MPI_Comm_size(MPI_COMM_WORLD, &psize);
MPI_Comm_rank(MPI_COMM_WORLD, &prank);
  }

~environment() 
  { MPI_Finalize(); }

environment& operator=(const environment&) = delete;
environment& operator=(environment&&)      = delete;

/// @brief Return the number of processes in @c MPI_COMM_WORLD
Integer size();

/// @brief Return the rank of current process in @c MPI_COMM_WORLD
Integer rank();

/// @brief Return the @c MPI_COMM_WORLD
MPI_Comm comm();

private:
Integer prank, psize;
MPI_Comm comm_world {MPI_COMM_WORLD};

}; 

} // end namespace mpi
} // end namespace final_project



/// 
/// Definitions of inline-member functions
///

namespace final_project { namespace mpi {

Integer 
  inline 
  environment::size()
  { return psize; }

Integer 
  inline 
  environment::rank()
  { return prank; }

MPI_Comm
  inline 
  environment::comm()
  { return comm_world; }

} // end namespace mpi
} // end namespace final_project



#endif // end of define FINAL_PROJECT_ENVRIONMENT_HPP_LIYIHAI










