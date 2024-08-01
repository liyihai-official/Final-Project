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
    template <typename ... Args>
    Heat_Base(mpi::environment &, Args ...);

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
    virtual void update_ping_pong()           = 0;
    virtual void update_ping_pong_bulk()      = 0;
    virtual void update_ping_pong_boundary()  = 0;
    
  }; 

template <typename T>
  class Heat_2D : protected Heat_Base<T, 2> 
  {

    friend class BoundaryConditions::DirchletBC<T, 2>;
    friend class BoundaryConditions::NeumannBC<T, 2>;

    public:
    Heat_2D(mpi::environment & env, size_type nx, size_type ny)
  : Heat_Base<T, 2>(env, nx, ny)
    {
      exchange_ping_pong();
    }

    
    void exchange_ping_pong() override
    {
      std::cout << " EXCHANGE PING PONG " << this->dt << std::endl;
    }

    void update_ping_pong() override
    {
      std::cout << "UPDATE" << std::endl;
    }
    void update_ping_pong_bulk() override 
    {

    }

    void update_ping_pong_boundary() override  
    {

    }

    private:
    // mpi::array_Cart<T, 2>();
  
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
template <typename ... Args>
  inline
  Heat_Base<T, NumD>::Heat_Base(mpi::environment & env, Args ... args)
  {
    std::cout << "Constructor from Base" << std::endl;

    multi_array::__detail::__multi_array_shape<NumD> shape(args...);

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


} // namespace pde
} // namespace final_project



#endif // end of define FINAL_PROJECT_HEAT_HPP