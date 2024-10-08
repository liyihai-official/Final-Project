
#ifndef HEAT_HPP
#define HEAT_HPP

#pragma once
#include "array.hpp"
#include <algorithm>

namespace final_project {


template <typename T, std::size_t NumDims>
class heat_equation {

  public:
  template <typename ... Args>
  heat_equation(Args ... args)
  { 
    auto glob_shape {
      final_project::__detail::__types::__multi_array_shape<NumDims>(args ...)
    };

    std::fill(minRange.begin(), minRange.end(), 0);
    std::fill(maxRange.begin(), maxRange.end(), 1); 

    for (std::size_t i = 0; i < NumDims; ++i)
    {
      dxs[i] = (maxRange[i] - minRange[i]) / glob_shape[i];
    }

    auto mmadr {std::min_element(dxs.begin(), dxs.end())};
    dt = *mmadr * *mmadr / coff / std::pow(2, NumDims);
    dt = std::min(dt, 0.1);

    for (std::size_t i = 0; i < NumDims; ++i)
    {
      weights[i] = coff * dt / (dxs[i] * dxs[i]);
      diags[i] = -2.0 + (dxs[i] * dxs[i]) / (NumDims * coff * dt);
    }
  }

  public:
  T coff {1};
  T dt {0.1}; 
  std::array<T, NumDims> minRange, maxRange;
  std::array<T, NumDims> diags, weights, dxs;

  // public:
  // void DirchletBoundaryCondition();
  // void RobbinBoundaryCondition();
  // void VonNeummanBoundaryCondition();

  // public:
  // void exchange_ping_pong1(final_project::array::array_distribute<T, NumDims> &)

  // public:
  // void update_ping_pong1( final_project::array::array_distribute<T, NumDims> & , 
  //                         final_project::array::array_distribute<T, NumDims> &)
};






} // namespace final_project


#endif // define HEAT_HPP