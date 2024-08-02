#ifndef FINAL_PROJECT_HEAT_2D_HPP
#define FINAL_PROJECT_HEAT_2D_HPP

#pragma once
#include <pde/Heat.hpp>


namespace final_project {  namespace pde {


template <typename T>
  class Heat_2D : protected Heat_Base<T, 2> 
  {
    public:
    Heat_2D(mpi::environment &, size_type, size_type);

    Integer solve_pure_mpi(T, Integer=100, Integer=0);

    private:
    mpi::array_Cart<T, 2> in, out;
    multi_array::array_base<T, 2> gather;

    private:
    void exchange_ping_pong() override;

    T update_ping_pong() override;
    T update_ping_pong_bulk() override;
    T update_ping_pong_boundary() override;
    void switch_in_out();

    friend InitialConditions::Init_2D<T>;
    friend BoundaryConditions_2D<T>;
  
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
    gather = multi_array::array_base<T, 2>(nx, ny);

    in.array().__loc_array.fill(0);
    out.array().__loc_array.fill(0);    
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
      for ( j=1; j < in.topology().__local_shape[1] - 1; ++j)
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
    Integer flag {0}; size_type dim {0};
    
    auto n_size {in.topology().__local_shape[dim]};
    MPI_Sendrecv( &in(1 ,1), 1, in.topology().halos[dim], in.topology().nbr_src[dim], flag,
                  &in(n_size-1 ,1), 1, in.topology().halos[dim], in.topology().nbr_dest[dim] , flag,
                  in.topology().comm_cart, MPI_STATUS_IGNORE);

    MPI_Sendrecv( &in(n_size-2 ,1), 1, in.topology().halos[dim], in.topology().nbr_dest[dim], flag,
                  &in(0 ,1)       , 1, in.topology().halos[dim], in.topology().nbr_src[dim] , flag,
                  in.topology().comm_cart, MPI_STATUS_IGNORE);

    flag = 1; dim = 1;
    n_size = in.topology().__local_shape[dim];
    MPI_Sendrecv( &in(1,        1), 1, in.topology().halos[dim], in.topology().nbr_src[dim], flag,
                  &in(1, n_size-1), 1, in.topology().halos[dim], in.topology().nbr_dest[dim] , flag,
                  in.topology().comm_cart, MPI_STATUS_IGNORE);

    MPI_Sendrecv( &in(1, n_size-2), 1, in.topology().halos[dim], in.topology().nbr_dest[dim], flag,
                  &in(1, 0)       , 1, in.topology().halos[dim], in.topology().nbr_src[dim] , flag,
                  in.topology().comm_cart, MPI_STATUS_IGNORE);
  }


template <typename T>
  Integer 
  Heat_2D<T>::solve_pure_mpi(T tol, Integer nsteps, Integer root)
    {
      T ldiff {0.0}, gdiff {0.0}; MPI_Datatype DiffType {mpi::get_mpi_type<T>()};
      Bool converge {false}; Integer iter;

      for (iter = 1; iter < nsteps; ++iter)
      {
        exchange_ping_pong();
        ldiff = update_ping_pong();
        MPI_Allreduce(&ldiff, &gdiff, 1, DiffType, MPI_SUM, in.topology().comm_cart);

        if (in.topology().rank == root) 
          std::cout << std::fixed << std::setprecision(13) << std::setw(15) << gdiff << std::endl;

        if (gdiff  <= tol) {
          converge = true;
          break;
        }

        switch_in_out();
      }

      mpi::Gather(gather, in, root);
      if (in.topology().rank == root)
        std::cout << gather.data() << std::endl;

      return iter;
    }  
} // namespace pde
} // namespace final_project



#endif // end of define FINAL_PROJECT_HEAT_2D_HPP