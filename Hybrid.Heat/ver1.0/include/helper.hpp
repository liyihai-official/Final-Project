///
/// @file helper.hpp
/// @brief
/// @author LI Yihai
///

#ifndef FINAL_PROJECT_HELPER_HPP_LIYIHAI
#define FINAL_PROJECT_HELPER_HPP_LIYIHAI

#pragma once
#include <types.hpp>
#include <mpi/environment.hpp>
#include <iostream>

namespace final_project 
{

  // Parallel Strategy
  enum class Strategy {
    PURE_MPI,
    HYBRID_0,
    HYBRID_1,
    UNKNOWN
  };


  Strategy getStrategyfromString(const String&); 
  std::ostream& operator<<(std::ostream&, Strategy);

  void helper_message(mpi::environment &);
  void version_message(mpi::environment &);
  
} // namespace final_project



#endif // end define FINAL_PROJECT_HELPER_HPP_LIYIHAI