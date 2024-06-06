

#ifndef FINAL_PROJECT_TYPES_HPP_LIYIHAI
#define FINAL_PROJECT_TYPES_HPP_LIYIHAI

#include <mpi.h>
#include <memory>

template<typename T>
MPI_Datatype get_mpi_type();

template<>
MPI_Datatype get_mpi_type<int>()    { return MPI_INT; }

template<>
MPI_Datatype get_mpi_type<float>()  { return MPI_FLOAT; }

template<>
MPI_Datatype get_mpi_type<double>() { return MPI_DOUBLE; }

namespace final_project
{
  namespace _detail 
  {

    typedef std::size_t _size_type;

    

    template <typename T>
    struct _mpi_topology
    {
      int rank, num_proc, dimension;
      std::unique_ptr<int[]> neighbors, starts, ends, coordinates;

      MPI_Comm comm;
      MPI_Datatype type{get_mpi_type<T>()};
    };


    template <std::size_t NumDims>
    struct _multi_array_shape 
    {
      typedef final_project::_detail::_size_type _size_type;

      std::unique_ptr<_size_type[]> sizes;

      _multi_array_shape() : sizes(std::make_unique<_size_type[]>(NumDims)) {}

      _size_type& operator[] (_size_type index) { return sizes[index]; }

      const _size_type& operator[] (_size_type index) const { return sizes[index]; }

      const _size_type dim() const { return NumDims; }

      const _size_type size() const {
        _size_type total {1} ;
        for (_size_type i = 0; i < NumDims; ++i) total *= sizes[i];
        return total;
      }
    };



  }
}
#endif