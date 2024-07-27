///
/// @file mpi/assert.hpp
/// @brief Defines the MPI assert using `assert`
/// @autor LI Yihai
/// @version 6.0
/// 
#ifndef FINAL_PROJECT_MPI_ASSERT_HPP_LIYIHAI
#define FINAL_PROJECT_MPI_ASSERT_HPP_LIYIHAI

///
/// FINAL_PROJECT_MPI_WARN
/// FINAL_PROJECT_MPI_ASSERT, FINAL_PROJECT_ASSERT_GLOBAL
///
#if defined(FINAL_PROJECT_DISABLE_ASSERTS) || defined(NDEBUG)

#define FINAL_PROJECT_MPI_WARN(expr) ((void)0)
#define FINAL_PROJECT_MPI_ASSERT(expr) ((void)0)
#define FINAL_PROJECT_MPI_ASSERT_GLOBAL(expr) ((void)0)
#define FINAL_PROJECT_MPI_INIT_CHECK() ((void)0)
#define FINAL_PROJECT_MPI_FINALIZE_CHECK() ((void)0)
#define FINAL_PROJECT_MPI_ASSERT_IS_VOID

#else

#pragma once
#include <mpi.h>
#include <cassert>
#include <iostream>

#define FINAL_PROJECT_MPI_WARN(expr) \
  if (!(expr)) { \
    int rank, size; \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
    MPI_Comm_size(MPI_COMM_WORLD, &size); \
    std::cerr << "MPI Assertion Failed: " #expr \
              << ", on rank " << rank << "/" << size \
              << ", file " << __FILE__ \
              << ", line " << __LINE__ \
              << std::endl; \
  }

#define FINAL_PROJECT_MPI_ASSERT(expr) \
  do { \
    int cond = (expr); \
    if (!cond) { \
      FINAL_PROJECT_MPI_WARN(cond); \
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); \
    } \
  } while (0)

#define FINAL_PROJECT_MPI_ASSERT_GLOBAL(expr) \
  do { \
    int loc_cond = (expr), glob_cond; \
    MPI_Allreduce(&loc_cond, &glob_cond, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD); \
    if (!glob_cond) { \
      FINAL_PROJECT_MPI_WARN(loc_cond); \
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); \
    } \
  } while (0)


#define FINAL_PROJECT_MPI_INIT_CHECK() \
  { \
    int initialized; \
    MPI_Initialized(&initialized); \
    FINAL_PROJECT_MPI_ASSERT(initialized); \
  }


#define FINAL_PROJECT_MPI_FINALIZE_CHECK() \
  { \
    int finalized; \
    MPI_Finalized(&finalized); \
    FINAL_PROJECT_MPI_ASSERT(finalized); \
  }


// #define FINAL_PROJECT_SEND()
// #define FINAL_PROJECT_RECV()



#if defined(NDEBUG)
#define FINAL_PROJECT_MPI_ASSERT_IS_VOID
#endif

#endif // end of define

#endif // end of define FINAL_PROJECT_MPI_ASSERT_HPP_LIYIHAI