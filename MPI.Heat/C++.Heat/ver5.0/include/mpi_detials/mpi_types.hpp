///
/// @file mpi_types.hpp
/// @brief This short file contains the details of template MPI Basic Datatypes
///
/// @author LI Yihai
/// @version 5.0 
/// @date Jun 26, 2024

#pragma once 
#include <mpi.h>



// ------------------------------- Header File ------------------------------- // 

namespace final_project {
namespace __detail {
namespace __mpi_types {

/// @brief Retrieve the MPI datatype for a given template type.
///
/// @tparam T the data type.
/// @return MPI_Datatype The MPI type correspond for T data type.
template<typename T>
MPI_Datatype __get_mpi_type();

} // namespace __mpi_types
} // namespace __detail
} // namespace final_project



// ------------------------------- Source File ------------------------------- // 

namespace final_project {
namespace __detail {
namespace __mpi_types {

template<>
MPI_Datatype __get_mpi_type<int>()    { return MPI_INT; }
 
template<>
MPI_Datatype __get_mpi_type<float>()  { return MPI_FLOAT; }

template<>
MPI_Datatype __get_mpi_type<double>() { return MPI_DOUBLE; }


} // namespace __mpi_types
} // namespace __detail
} // namespace final_project