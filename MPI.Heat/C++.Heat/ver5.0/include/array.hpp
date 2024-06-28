


#include <memory>
#include "mpi_distribute/mpi_distribute_array.hpp"



namespace final_project {
namespace array {


template <class T, std::size_t NumDims>
  class array_distribute {
    
    private:
    typedef T                                             value_type;
    typedef __detail::__mpi_distribute_array<T, NumDims>  array_type;

    typedef __detail::__mpi_types::__size_type                size_type;
    typedef __detail::__types::__multi_array_shape<NumDims>   super_array_shape;

    typedef mpi::env                                          mpi_env;
    typedef __detail::__mpi_types::__mpi_topology<T, NumDims> mpi_topology;
    
    private:
    std::unique_ptr<array_type> body;

    public:
    array_distribute(super_array_shape & global_shape, mpi_env & env)
    : body(std::make_unique<array_type>(global_shape, env)) {}

    public:

}; // class array_distribute


} // array
} // namespace final_project