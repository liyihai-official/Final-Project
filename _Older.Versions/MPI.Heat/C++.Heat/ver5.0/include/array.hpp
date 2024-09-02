#include <memory>
#include <vector>
#include <fstream>

#pragma once
#include "mpi_distribute/mpi_distribute_array.hpp"
#include "assert.hpp"

#include "fdm/heat.hpp"

namespace final_project {
namespace array {

/// @brief 
/// @tparam T 
/// @tparam NumDims 
template <class T, std::size_t NumDims>
  class array_base {
    friend final_project::heat_equation<T, NumDims>;

    private:
    typedef T                                               value_type;
    typedef __detail::__multi_array::__array<T, NumDims>    array_type;

    typedef __detail::__mpi_types::__size_type              size_type;
    typedef __detail::__types::__multi_array_shape<NumDims> super_array_shape;

    private:
    std::unique_ptr<array_type> body;

    public:
    array_base(super_array_shape & array_base_shape)
    : body (std::make_unique<array_type>(array_base_shape)) 
    { FINAL_PROJECT_ASSERT_MSG((NumDims < 4), "Invalid Dimension of Array.\n"); }

    template <typename ... Args>
    array_base( Args ... args )
    : body(std::make_unique<array_type>(super_array_shape(args ...)))
    { FINAL_PROJECT_ASSERT_MSG((NumDims < 4), "Invalid Dimension of Array.\n"); }

    public:
    array_type& get_array()
    { return *body; }

    array_type& get_array() const
    { return *body; }

    size_type& shape(size_type index) const
    { return body->__shape[index]; }

    void saveToBinaryFile(const std::string & filename) const {
      std::ofstream ofs(filename, std::ios::binary);

      if (!ofs) { throw std::runtime_error("Cannot Open File."); }

      for (size_type i = 0; i < NumDims; ++i) {
        auto temp {body->__shape[i]};
        ofs.write(reinterpret_cast<const char*>(&temp), sizeof((temp)));
      }
      
      ofs.write(reinterpret_cast<const char*>(body->begin()), body->size() * sizeof(T));
      
    }

 };

/// @brief 
/// @tparam T 
/// @tparam NumDims 
template <class T, std::size_t NumDims>
  class array_distribute {
    
    private:
    typedef T                                                 value_type;
    typedef __detail::__mpi_distribute_array<T, NumDims>      array_type;

    typedef __detail::__mpi_types::__size_type                size_type;
    typedef __detail::__types::__multi_array_shape<NumDims>   super_array_shape;

    typedef mpi::env                                          mpi_env;
    typedef __detail::__mpi_types::__mpi_topology<T, NumDims> mpi_topology;
    
    private:
    std::unique_ptr<array_type> body;

    public:
    array_distribute() = default;
    array_distribute(super_array_shape &, mpi_env &);

    template <typename ... Args>
    array_distribute(mpi_env & env, Args ... args)
    : body([&]() 
    {
      super_array_shape shape(args...);
      return std::make_unique<array_type>(shape, env);
    }())
    { FINAL_PROJECT_MPI_ABORT_IF_FALSE((NumDims < 4), env.comm(), 1, "Invalid Dimension of Array.\n"); }

    public:
    void swap(array_distribute & other) 
    { body.swap(other.body); }

    array_type& get_array();
    array_type& get_array() const;

    mpi_topology& get_topology() const 
    { return body->__local_topology; }

    public:
    void fill_boundary(const T);

}; // class array_distribute

} // namespace array

} // namespace final_project





namespace final_project {
namespace array {

template <class T, std::size_t NumDims>
  inline 
  array_distribute<T, NumDims>::array_distribute(super_array_shape & global_shape, mpi_env & env)
  : body(std::make_unique<array_type>(global_shape, env)) {
FINAL_PROJECT_MPI_ABORT_IF_FALSE((NumDims < 4), env.comm(), 1, "Invalid Dimension of Array.\n");
  }

template <class T, std::size_t NumDims>
  inline 
  typename array_distribute<T, NumDims>::array_type& 
  array_distribute<T, NumDims>::get_array() 
  { return *body; }

template <class T, std::size_t NumDims>
  inline 
  typename array_distribute<T, NumDims>::array_type&
  array_distribute<T, NumDims>::get_array() const 
  { return *body; }

template <class T, std::size_t NumDims>
  inline 
  void array_distribute<T, NumDims>::fill_boundary(const T junk_value)
  {
    if (NumDims == 1)
    {
      if (body->__local_topology.__starts[0] == 1) 
        body->__local_array(0) = junk_value;

      if (body->__local_topology.__ends[0] == body->__local_topology.__global_shape[0] - 2) 
        body->__local_array(body->__local_array.__shape[0]-1) = junk_value;
    }

    if (NumDims == 2)
    {
      std::size_t i, j;
      auto ni {body->__local_array.__shape[0]},             nj {body->__local_array.__shape[1]};
      auto Ni {body->__local_topology.__global_shape[0]},   Nj {body->__local_topology.__global_shape[1]};

      if (body->__local_topology.__starts[0] == 1)
        for (j = 0; j < nj; ++j) body->__local_array(0, j) = junk_value;
      
      if (body->__local_topology.__starts[1] == 1)
        for (i = 0; i < ni; ++i) body->__local_array(i, 0) = junk_value * 0.1;

      if (body->__local_topology.__ends[0] == Ni - 2)
        for (j = 0; j < nj; ++j) body->__local_array(ni-1, j) = junk_value * 0.3;

      if (body->__local_topology.__ends[1] == Nj - 2)
        for (i = 0; i < ni; ++i) body->__local_array(i, nj - 1) = junk_value * 0.5; 
    }

    if (NumDims == 3)
    {
      std::size_t i, j, k;
      auto  ni {body->__local_array.__shape[0]},             
            nj {body->__local_array.__shape[1]},            
            nk {body->__local_array.__shape[2]};
      auto  Ni {body->__local_topology.__global_shape[0]},   
            Nj {body->__local_topology.__global_shape[1]},  
            Nk {body->__local_topology.__global_shape[2]};

      if (body->__local_topology.__starts[0] == 1) 
        for (j = 0; j < nj; ++j) 
          for (k = 0; k < nk; ++k) body->__local_array(0,j,k) = junk_value * 0.1;

      if (body->__local_topology.__starts[1] == 1)
        for (i = 0; i < ni; ++i)
          for (k = 0; k < nk; ++k) body->__local_array(i,0,k) = junk_value * 0.3;

      if (body->__local_topology.__starts[2] == 1)
        for (i = 0; i < ni; ++i)
          for (j = 0; j < nj; ++j) body->__local_array(i,j,0) = junk_value;

      if (body->__local_topology.__ends[0] == Ni - 2)
        for (j = 0; j < nj; ++j) 
          for (k = 0; k < nk; ++k) body->__local_array(ni-1,j,k) = junk_value * 0.7;

      if (body->__local_topology.__ends[1] == Nj - 2)
        for (i = 0; i < ni; ++i)
          for (k = 0; k < nk; ++k) body->__local_array(i,nj-1,k) = junk_value * 0.8;

      if (body->__local_topology.__ends[2] == Nk - 2)
        for (i = 0; i < ni; ++i)
          for (j = 0; j < nj; ++j) body->__local_array(i,j,nk-1) = junk_value * 0.5;
    }
  }






/// @brief 
/// @tparam T 
/// @tparam NumDim 
/// @param gather 
/// @param array 
template <typename T, std::size_t NumDim>
void Gather(
  array_base<T, NumDim>       & gather,   
  array_distribute<T, NumDim> & array)
{
  MPI_Datatype sbuf_block, temp, mpi_T {final_project::__detail::__mpi_types::__get_mpi_type<T>()};

  int root {0};

  int indexs[NumDim];
  
  int num_procs {array.get_topology().__num_procs};
  int pid, i, j, k;

  int Ns[NumDim], starts_cpy[NumDim];

  int s_list[NumDim][num_procs], n_list[NumDim][num_procs];

  int array_of_sizes[NumDim], array_of_starts[NumDim], array_of_subsizes[NumDim];

  for (std::size_t dim = 0; dim < NumDim; ++dim)
  {
    indexs[dim] = 1;
    Ns[dim] = array.get_topology().__ends[dim] - array.get_topology().__starts[dim] + 1; 
    starts_cpy[dim] = array.get_topology().__starts[dim];

    if (starts_cpy[dim] == 1) 
    {
      -- starts_cpy[dim];
      -- indexs[dim];
      ++ Ns[dim];
    }

    if (array.get_topology().__ends[dim] == array.get_topology().__global_shape[dim] - 2) 
      ++ Ns[dim];
    
    MPI_Gather(&starts_cpy[dim], 1, MPI_INT, s_list[dim], 1, MPI_INT, root, array.get_topology().__comm_cart);
    MPI_Gather(&Ns[dim], 1, MPI_INT, n_list[dim], 1, MPI_INT, root, array.get_topology().__comm_cart);
  }

  if (array.get_topology().__rank != root)
  {
    for (std::size_t dim = 0; dim < NumDim; ++dim)
    {
      array_of_sizes[dim] = array.get_topology().__local_shape[dim];
      array_of_subsizes[dim] = Ns[dim];
      array_of_starts[dim] = 0;
    }

    MPI_Type_create_subarray( array.get_topology().__dimension, 
                              array_of_sizes,
                              array_of_subsizes,
                              array_of_starts,
                              MPI_ORDER_C, mpi_T, &sbuf_block);

    MPI_Type_commit(&sbuf_block);

    if (NumDim == 2)
      MPI_Send( &(array.get_array().__local_array(indexs[0], indexs[1])), 
                1, sbuf_block, root, 
                array.get_topology().__rank, 
                array.get_topology().__comm_cart);

    if (NumDim == 3)
      MPI_Send( &(array.get_array().__local_array(indexs[0], indexs[1], indexs[2])), 
                1, sbuf_block, root, 
                array.get_topology().__rank, 
                array.get_topology().__comm_cart);

    MPI_Type_free(&sbuf_block);
  }


  if (array.get_topology().__rank == root)
  {
    for (pid = 0; pid < num_procs; ++pid)
    {
      if (pid == root)
      {
        for ( i=starts_cpy[0]; i <= array.get_topology().__ends[0]; ++i) 
        {
          if (NumDim == 2) 
            memcpy( &gather.get_array()(i, starts_cpy[1]), 
                    &array.get_array().__local_array(i, starts_cpy[1]), n_list[1][pid]*sizeof(T));

          if (NumDim == 3) {
            for ( j=starts_cpy[1]; j <= array.get_topology().__ends[1]; ++j)
              memcpy( &gather.get_array()(i, j, starts_cpy[2]), 
                      &array.get_array().__local_array(i, j, starts_cpy[2]), n_list[2][pid]*sizeof(T));
          }
        }        
      }

      if (pid != root) 
      {
        for (std::size_t dim = 0; dim < NumDim; ++dim)
        {
          array_of_starts[dim] = 0;
          array_of_sizes[dim] = array.get_topology().__global_shape[dim];
          array_of_subsizes[dim] = n_list[dim][pid];
        }

        MPI_Type_create_subarray( array.get_topology().__dimension, 
                                  array_of_sizes, 
                                  array_of_subsizes,
                                  array_of_starts,
                                  MPI_ORDER_C, mpi_T, &temp);

        MPI_Type_commit(&temp);

        // s_list[NumDim][num_procs]
        if (NumDim == 2)
          MPI_Recv( &gather.get_array()(s_list[0][pid], s_list[1][pid]), 
                    1, temp, pid, pid, 
                    array.get_topology().__comm_cart,
                    MPI_STATUS_IGNORE);

        if (NumDim == 3)
          MPI_Recv( &gather.get_array()(s_list[0][pid], s_list[1][pid], s_list[2][pid]), 
                    1, temp, pid, pid, 
                    array.get_topology().__comm_cart,
                    MPI_STATUS_IGNORE);

        MPI_Type_free(&temp);
      }
    }
  }

}


} // array
} // namespace final_project