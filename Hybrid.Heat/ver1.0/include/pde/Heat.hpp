///
/// @file pde/Heat.hpp
///
#ifndef FINAL_PROJECT_HEAT_HPP
#define FINAL_PROJECT_HEAT_HPP

#pragma once

// Final Project
#include <pde/pde.hpp>


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
    virtual void exchange_ping_pong_SR()      = 0;
    virtual void exchange_ping_pong_I()       = 0;

    // Updates
    public: 
    virtual T update_ping_pong()           = 0;
    virtual T update_ping_pong_bulk()      = 0;
    virtual T update_ping_pong_boundary()  = 0;
      
  }; 


// template <typename T>
//   class Heat_3D : public Heat_Base<T, 3> 
//    { }; // class Heat<T, 3>

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
    #ifndef NDEBUG
    std::cout << "Constructor from Heat_Base" << std::endl;
    #endif

    multi_array::__detail::__multi_array_shape<NumD> shape(exts...);

    std::fill(minRange.begin(), minRange.end(), 0);
    std::fill(maxRange.begin(), maxRange.end(), 1);

    for (std::size_t i = 0; i < NumD; ++i)
    {
      dxs[i] = (maxRange[i] - minRange[i]) / (shape[i]-1);
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