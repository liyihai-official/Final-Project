#ifndef FINAL_PROJECT_INITIALIZATIONS_C_HPP
#define FINAL_PROJECT_INITIALIZATIONS_C_HPP

#pragma once 
#include <functional>

#include <pde/detials/Heat_2D.hpp>


namespace final_project { namespace pde {

namespace InitialConditions {


template <typename T>
  class Init_2D
  {
    using InitFunction = std::function<T(T, T)>;

    public:
    Init_2D();
    Init_2D(Heat_2D<T> &, InitFunction );

    private:
    InitFunction initFunc;
  }; // class Init_2D

///
///
/// Inline Function Definitions
///
///


template <typename T>
  Init_2D<T>::Init_2D()
  : initFunc([](T x, T y) { return 0; })
  {}

template <typename T>
  Init_2D<T>::Init_2D(Heat_2D<T> & eqs, InitFunction initFunc)
  : initFunc {initFunc}
  {
    T x{0}, y{0};
    for (size_type i = 1; i < eqs.in.topology().__local_shape[0]-1; ++i)
    {
      for (size_type j = 1; j < eqs.in.topology().__local_shape[1]-1; ++j)
      {
        x = (i) * eqs.dxs[0];
        y = (j) * eqs.dxs[1];

        eqs.in(i,j) = initFunc(x,y);
        eqs.out(i,j) = initFunc(x,y);
      }
    }

  }


} // namespace InitialConditions





} // namespace pde
} // namespace final_project







#endif // define FINAL_PROJECT_INITIALIZATIONS_C_HPP