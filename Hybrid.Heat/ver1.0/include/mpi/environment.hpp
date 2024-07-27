/// 
/// 
/// 
///


#ifndef FINAL_PROJECT_ENVRIONMENT_HPP_LIYIHAI
#define FINAL_PROJECT_ENVRIONMENT_HPP_LIYIHAI

#pragma once
#include <mpi.h>

#include "types.hpp"

#include "assert.hpp"
#include "mpi/assert.hpp"

namespace final_project { namespace mpi {

class environment {

environment(int& argc, char** &argv, int required = MPI_THREAD_SINGLE)
{
  int provided;
  MPI_Init_thread(&argc, &argv, required, &provided);
  FINAL_PROJECT_MPI_INIT_CHECK();
}

~environment() { MPI_Finalize(); }

}; 


} // end namespace mpi
} // end namespace final_project



#endif // end of define FINAL_PROJECT_ENVRIONMENT_HPP_LIYIHAI










