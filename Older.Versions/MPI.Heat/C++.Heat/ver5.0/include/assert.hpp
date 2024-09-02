/**
 * @file assert.hpp
 * @brief This file defines a macro for custom assertions with msgs.
 * 
 * This header file provides a macro FINAL_PROJECT_ASSERT_MSG that can 
 * be used for assertions with custom msgs. It is based on the 
 * standard assert macro.
 * 
 * @date Jun 26, 2024
 * @version 5.0
 */

#ifndef FINAL_PROJECT_ASSERT_LIYIHAI
#define FINAL_PROJECT_ASSERT_LIYIHAI

#include <cassert>
#include <source_location>
#include <string>
#include <mpi.h>

#define FINAL_PROJECT_ASSERT_MSG(expr, msg) assert((expr) && (msg))

// std::string sourceline(const std::source_location location)
// {
//     auto line {location.line()};
//     auto column {location.column()};
//     std::string result {"file: "};
//     result += location.file_name();
//     result += "(" + std::to_string(line) + ":" + std::to_string(column) + ")";
//     result += " '";
//     result += location.function_name();
//     return result;   
// }

#define FINAL_PROJECT_MPI_ABORT_IF_FALSE(expr, comm, errorcode, msg) \
  do { \
      if (!(expr)) { \
  int rank; \
  MPI_Comm_rank(comm, &rank); \
  fprintf(stderr, "Error in process %d: %s\n", rank, msg); \
  MPI_Abort(comm, errorcode); \
      } \
  } while (0)



#endif