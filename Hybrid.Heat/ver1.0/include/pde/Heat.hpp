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
    protected:
    template <typename ... Exts>
    Heat_Base(Exts ...);

    protected:
    // Coefficients of Heat Equation
    T coff {2}, dt {0.1};
    std::array<T, NumD> minRange, maxRange;
    std::array<T, NumD> diags, weights, dxs;

    // Exchange 
    protected:
    virtual void exchange_ping_pong_SR()      = 0;
    virtual void exchange_ping_pong_I()       = 0;

    // Updates
    protected: 
    virtual T update_ping_pong()                = 0;
    virtual T update_ping_pong_omp()            = 0;
    virtual T update_ping_pong_bulk()           = 0;
    virtual T update_ping_pong_edge()           = 0;


    virtual ~Heat_Base() {};
  }; 

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
  Heat_Base<T, NumD>::Heat_Base(Exts ... exts)
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
    dt = *mmadr * *mmadr / coff / (2 * NumD ); //std::pow(2, NumD));
    T mindt {0.1};
    dt = std::min(dt, mindt);

    for (std::size_t i = 0; i < NumD; ++i)
    {
      weights[i] = coff * dt / (dxs[i] * dxs[i]);
      diags[i] = -2.0 + (dxs[i] * dxs[i]) / (NumD * coff * dt);
    }
    
  }
} // namespace pde
} // namespace final_project



#endif // end of define FINAL_PROJECT_HEAT_HPP