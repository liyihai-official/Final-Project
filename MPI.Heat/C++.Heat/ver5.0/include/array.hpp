


#include <memory>
#include <vector>
#include <fstream>
#include "mpi_distribute/mpi_distribute_array.hpp"
#include "assert.hpp"

namespace final_project {
namespace array {

/// @brief 
/// @tparam T 
/// @tparam NumDims 
template <class T, std::size_t NumDims>
 class array_base {
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

  public:
  array_type& get_array()
  { return *body; }

  array_type& get_array() const
  { return *body; }

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
    typedef T                                             value_type;
    typedef __detail::__mpi_distribute_array<T, NumDims>  array_type;

    typedef __detail::__mpi_types::__size_type                size_type;
    typedef __detail::__types::__multi_array_shape<NumDims>   super_array_shape;

    typedef mpi::env                                          mpi_env;
    typedef __detail::__mpi_types::__mpi_topology<T, NumDims> mpi_topology;
    
    private:
    std::unique_ptr<array_type> body;

    public:
    array_distribute(super_array_shape &, mpi_env &);


    public:
    void swap(array_distribute & other) 
    {

      // FINAL_PROJECT_ASSERT_MSG((body->__local_topology == other.body->__local_topology), "Unmatched topology of distributed arrays.");
      // auto a = body->__local_topology == other.body->__local_topology;
      // std::cout << "CACA" << a << std::endl;
      body.swap(other.body); 
    }

    array_type& get_array();
    array_type& get_array() const;

    mpi_topology& get_topology() const 
    { return body->__local_topology; }

    public:
    void fill_boundary(const T);

    // T update()
    // {
    //   T diff {0}, temp;
    //   std::size_t off;

    //   if (NumDims == 2)
    //   {      
    //   #pragma omp parallel for private(temp) num_threads(2) reduction(+:diff)
    //   for (std::size_t id = 0; id < 2; ++id)
    //   {
    //     for (std::size_t i = 1; i < body->__local_array.__shape[0]-1; ++i)
    //     {
    //       off = 1 + (i + id + 1) % 2;
    //       for (std::size_t j = off; j < body->__local_array.__shape[1]-1; j+=2)
    //       {
    //         temp = body->__local_array(i,j);
    //         body->__local_array(i,j) = 0.25 * (
    //           body->__local_array(i-1,j) + body->__local_array(i+1,j) + 
    //           body->__local_array(i,j-1) + body->__local_array(i,j+1));

    //         diff += std::pow(temp - body->__local_array(i,j), 2);
    //       }
    //     }
    //   }
    //   diff = std::sqrt(diff / (T) ((body->__local_array.__shape[0]-1) * (body->__local_array.__shape[1]-1)));
    //   }

    // if (NumDims == 3)
    // {      
    //     #pragma omp parallel for private(temp) num_threads(2) reduction(+:diff)
    //     for (std::size_t id = 0; id < 2; ++id)
    //     {
    //         for (std::size_t i = 1; i < body->__local_array.__shape[0]-1; ++i)
    //         {
    //             for (std::size_t j = 1; j < body->__local_array.__shape[1]-1; ++j)
    //             {
    //                 for (std::size_t k = 1; k < body->__local_array.__shape[2]-1; ++k)
    //                 {
    //                     if ((i + j + k + id) % 2 == 0)
    //                     {
    //                       temp = body->__local_array(i,j,k);
    //                       body->__local_array(i,j,k) = (1.0 / 6.0) * (
    //                           body->__local_array(i-1,j,k) + body->__local_array(i+1,j,k) +
    //                           body->__local_array(i,j-1,k) + body->__local_array(i,j+1,k) +
    //                           body->__local_array(i,j,k-1) + body->__local_array(i,j,k+1));

    //                       diff += std::pow(temp - body->__local_array(i,j,k), 2);
    //                     }
    //                 }
    //             }
    //         }
    //     }
    //     diff = std::sqrt(diff / (T) (
    //         (body->__local_array.__shape[0]-2) * 
    //         (body->__local_array.__shape[1]-2) * 
    //         (body->__local_array.__shape[2]-2)
    //     ));
    // }
    //   return diff;
    // }

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
          for (k = 0; k < nk; ++k) body->__local_array(0,j,k) = junk_value;

      if (body->__local_topology.__starts[1] == 1)
        for (i = 0; i < ni; ++i)
          for (k = 0; k < nk; ++k) body->__local_array(i,0,k) = junk_value;

      if (body->__local_topology.__starts[2] == 1)
        for (i = 0; i < ni; ++i)
          for (j = 0; j < nj; ++j) body->__local_array(i,j,0) = junk_value;

      if (body->__local_topology.__ends[0] == Ni - 2)
        for (j = 0; j < nj; ++j) 
          for (k = 0; k < nk; ++k) body->__local_array(ni-1,j,k) = junk_value;

      if (body->__local_topology.__ends[1] == Nj - 2)
        for (i = 0; i < ni; ++i)
          for (k = 0; k < nk; ++k) body->__local_array(i,nj-1,k) = junk_value;

      if (body->__local_topology.__ends[2] == Nk - 2)
        for (i = 0; i < ni; ++i)
          for (j = 0; j < nj; ++j) body->__local_array(i,j,nk-1) = junk_value;
    }
  }









} // array
} // namespace final_project