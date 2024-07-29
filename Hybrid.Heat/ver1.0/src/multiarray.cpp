///
/// @file src/multiarray.hpp
/// @brief This source file provides the none-inline definitions of 
///       the objects and functions which prototypes are given in the 
///       file include/multiarray.hpp
///
/// @author LI Yihai
/// @date 2024 Jul. 28
/// @version 6.0
///


#ifndef FINAL_PROJECT_MULTI_ARRAY_CPP_LIYIHAI
#define FINAL_PROJECT_MULTI_ARRAY_CPP_LIYIHAI
#include <array>

#include <multiarray.hpp>

#include <assert.hpp>
#include <mpi/assert.hpp>

namespace final_project { namespace mpi {


  template <typename T, size_type NumD>
    void Gather(
      multi_array::array_base<T, NumD> & gather, 
      array_Cart<T, NumD> & loc,
      const Integer root=0)
    {
      MPI_Datatype sbuf_block, value_type {get_mpi_type<T>()};

      if (loc.topology().num_procs == 1) {
        *(gather.body).swap(*(loc.body).__loc_array);
      }

      Integer pid, i, j, k, dim, num_proc {loc.topology().num_procs}, dimension {loc.topology().dimension};
      Integer s_list[NumD][num_proc], n_list[NumD][num_proc];
      std::array<Integer, NumD> Ns, starts_cpy, array_sizes, array_subsizes, array_starts = {0}, indexes = {1};

      for (dim = 0; dim < dimension; ++dim)
      {
        Ns[dim] = loc.topology().__local_shape[dim] - 2;
        starts_cpy[dim] = loc.topology().starts[dim];

        if (starts_cpy[dim] == 1)
        {
          -- starts_cpy[dim];
          -- indexes[dim];
          ++ Ns[dim];
        }

        if (loc.topology().ends[dim] == loc.topology().__global_shape[dim] - 2)
          ++ Ns[dim];

        MPI_Datatype gtype {get_mpi_type<Integer>()};
        MPI_Gather(&starts_cpy[dim], 1, gtype, s_list[dim], 1, gtype, root, loc.topology().comm_cart);
        MPI_Gather(&Ns[dim], 1, gtype, n_list[dim], 1, gtype, root, loc.topology().comm_cart);  
      }

    }





  
} // namespace mpi
} // namespace final_project



#endif // end of define FINAL_PROJECT_MULTI_ARRAY_CPP_LIYIHAI

