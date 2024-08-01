///
/// @file pde/Heat.hpp
///
#ifndef FINAL_PROJECT_HEAT_HPP
#define FINAL_PROJECT_HEAT_HPP

#pragma once

// Final Project 
#include <types.hpp>
#include <pde/pde.hpp>
#include <multiarray.hpp>


namespace final_project {  namespace pde {


template <typename T, size_type NumD>
  class Heat_Base
  {
    public:
    template <typename ... Exts>
    Heat_Base(mpi::environment &, Exts ...);

    protected:
    // Coefficients of Heat Equation
    T coff {1}, dt {0.1};
    std::array<T, NumD> minRange, maxRange;
    std::array<T, NumD> diags, weights, dxs;


    // Exchange 
    public:
    virtual void exchange_ping_pong()         = 0;

    // Updates
    public: 
    virtual T update_ping_pong()           = 0;
    virtual T update_ping_pong_bulk()      = 0;
    virtual T update_ping_pong_boundary()  = 0;
    
  }; 

template <typename T>
  class Heat_2D : protected Heat_Base<T, 2> 
  {
    
    friend class BoundaryConditions::DirchletBC<T, 2>;
    friend class BoundaryConditions::NeumannBC<T, 2>;

    public:
    Heat_2D(mpi::environment & env, size_type nx, size_type ny);

    void exchange_ping_pong() override;

    T update_ping_pong() override;
    T update_ping_pong_bulk() override 
    {
      return 0;
    }
    T update_ping_pong_boundary() override  
    {
      return 0;
    }

    void show() 
    {
      std::cout << pong.array() << std::endl;
    }

    private:
    mpi::array_Cart<T, 2> ping, pong;
  
  }; // class Heat_2D



// template <typename T>
//   class Heat_3D : public Heat_Base<T, 3> {
//     public:


//   }; // class Heat<T, 3>

}}



///
/// Inline Function Definitions
///

// Standard Library
#include <cmath>
#include <algorithm>


namespace final_project { namespace pde {


template <typename T, size_type NumD>
template <typename ... Exts>
  inline
  Heat_Base<T, NumD>::Heat_Base(mpi::environment & env, Exts ... exts)
  {
    std::cout << "Constructor from Heat_Base" << std::endl;

    multi_array::__detail::__multi_array_shape<NumD> shape(exts...);

    std::fill(minRange.begin(), minRange.end(), 0);
    std::fill(maxRange.begin(), maxRange.end(), 1);

    for (std::size_t i = 0; i < NumD; ++i)
    {
      dxs[i] = (maxRange[i] - minRange[i]) / shape[i];
    }

    auto mmadr {std::min_element(dxs.begin(), dxs.end())};
    dt = *mmadr * *mmadr / coff / std::pow(2, NumD);
    dt = std::min(dt, 0.1);

    for (std::size_t i = 0; i < NumD; ++i)
    {
      weights[i] = coff * dt / (dxs[i] * dxs[i]);
      diags[i] = -2.0 + (dxs[i] * dxs[i]) / (NumD * coff * dt);
    }
  }


template <typename T>
  inline
  Heat_2D<T>::Heat_2D(mpi::environment & env, size_type nx, size_type ny)
: Heat_Base<T, 2>(env, nx, ny)
  {
    ping = mpi::array_Cart<T, 2>(env, nx, ny);
    pong = mpi::array_Cart<T, 2>(env, nx, ny);

    ping.array().__loc_array.fill(10);
    pong.array().__loc_array.fill(10);
    
  }


template <typename T>
  inline void 
  Heat_2D<T>::exchange_ping_pong()
  {

  }

template <typename T>
  inline T 
  Heat_2D<T>::update_ping_pong()
  {
    T diff {0.0};
    size_type i {1}, j {1};

    for (i=1; i < ping.topology().__local_shape[0] - 1; ++i)
    {
      for ( j = 1; j < ping.topology().__local_shape[1] - 1; ++j)
      {
        T current {ping(i,j)};
        pong(i,j) = 1 + 
            this->weights[0] * (ping(i-1, j) + ping(i+1,j))
          + this->weights[1] * (ping(i, j-1) + ping(i,j+1))
          + current * (
            this->diags[0]*this->weights[0] 
          + this->diags[1]*this->weights[1]);

        diff += std::pow(current - pong(i,j), 2);
      }
      
    }
    return diff;
  }

} // namespace pde
} // namespace final_project



#endif // end of define FINAL_PROJECT_HEAT_HPP