#ifndef FINAL_PROJECT_HEAT_2D_HPP
#define FINAL_PROJECT_HEAT_2D_HPP

#pragma once
#include <pde/Heat.hpp>



namespace final_project {  namespace pde {


template <typename T>
  class Heat_2D : protected Heat_Base<T, 2> 
  {
    
    friend class BoundaryConditions::DirchletBC<T, 2>;
    friend class BoundaryConditions::NeumannBC<T, 2>;

    public:
    Heat_2D(mpi::environment & env, size_type nx, size_type ny);

    void exchange_ping_pong() override;

    T update_ping_pong() override;
    T update_ping_pong_bulk() override;
    T update_ping_pong_boundary() override;


    void switch_in_out();
    void show() 
    {
      std::cout << in.array() << std::endl;
      // std::cout << out.array() << std::endl;
    }

    private:
    mpi::array_Cart<T, 2> in, out;
  
  }; // class Heat_2D


}}


///
/// Inline Function Definitions
///

// Standard Library
#include <cmath>
#include <algorithm>


namespace final_project { namespace pde {

template <typename T>
  inline
  Heat_2D<T>::Heat_2D(mpi::environment & env, size_type nx, size_type ny)
: Heat_Base<T, 2>(env, nx, ny)
  {
    in = mpi::array_Cart<T, 2>(env, nx, ny);
    out = mpi::array_Cart<T, 2>(env, nx, ny);


    // Boundary Conditions
    in.array().__loc_array.fill(0);
    out.array().__loc_array.fill(0);


    for (size_type i = 0; i < in.topology().__local_shape[0]; ++i)
    {

      size_type j = 0;
      // for (size_type j = 0; j < in.topology().__local_shape[1]; ++j)

      if (in.topology().rank == 0 || in.topology().rank == 1)
      {
        in(i,j) = 1;
        out(i,j) = 1;
      }
    }
    
  }

template <typename T>
  inline void 
  Heat_2D<T>::switch_in_out() 
{ in.swap(out); }

template <typename T>
  inline T 
  Heat_2D<T>::update_ping_pong()
  {
    T diff {0.0};
    size_type i {1}, j {1};

    for (i=1; i < in.topology().__local_shape[0] - 1; ++i)
    {
      for ( j = 1; j < in.topology().__local_shape[1] - 1; ++j)
      {
        T current {in(i,j)};
        out(i,j) =
            this->weights[0] * (in(i-1, j) + in(i+1,j))
          + this->weights[1] * (in(i, j-1) + in(i,j+1))
          + current * (
            this->diags[0]*this->weights[0] 
          + this->diags[1]*this->weights[1]);

        diff += std::pow(current - out(i,j), 2);
      }
      
    }
    return diff;
  }

template <typename T>
  inline T
  Heat_2D<T>::update_ping_pong_boundary()
  {
    return 0;
  }

template <typename T>
  inline T
  Heat_2D<T>::update_ping_pong_bulk()
  {
    return 0;
  }


template <typename T>
  inline void 
  Heat_2D<T>::exchange_ping_pong()
  {
    Integer flag {0};
    size_type dim {0};
    

    // in.topology().halos
    


// std::size_t dim {0};
// {
//   auto flag {dim};
//   auto n_size {out.get_array().__local_array.__shape[dim]};
//   MPI_Sendrecv( &out.get_array().__local_array(1,1), 1, 
//                 out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim], flag,
//                 &out.get_array().__local_array(n_size-1, 1), 1, 
//                 out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag,
//                 out.get_topology().__comm_cart, MPI_STATUS_IGNORE);

//   MPI_Sendrecv( &out.get_array().__local_array(n_size-2,1), 1, 
//                 out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag,
//                 &out.get_array().__local_array(0, 1), 1, 
//                 out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim], flag,
//                 out.get_topology().__comm_cart, MPI_STATUS_IGNORE);

//   dim = 1;
//   flag = dim;
//   n_size = out.get_array().__local_array.__shape[dim];
//   MPI_Sendrecv( &out.get_array().__local_array(1,        1), 1, 
//                 out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim  ], flag,
//                 &out.get_array().__local_array(1, n_size-1), 1, 
//                 out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag,
//                 out.get_topology().__comm_cart, MPI_STATUS_IGNORE);

//   MPI_Sendrecv( &out.get_array().__local_array(1, n_size-2), 1, 
//                 out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim+1], flag,
//                 &out.get_array().__local_array(1, 0), 1, 
//                 out.get_topology().__halo_vectors[dim], out.get_topology().__neighbors[2*dim], flag,
//                 out.get_topology().__comm_cart, MPI_STATUS_IGNORE);
// }
  }
} // namespace pde
} // namespace final_project



#endif // end of define FINAL_PROJECT_HEAT_2D_HPP