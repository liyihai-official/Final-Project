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
    Init_2D(InitFunction );

    private:
    InitFunction initFunc;
    void SetUpInit(Heat_2D<T> &);
    Bool isSetUpInit;

  
    friend Heat_2D<T>;
  }; // class Init_2D



///
///
/// Inline Function Definitions
///
///


template <typename T>
  Init_2D<T>::Init_2D()
  : initFunc([](T x, T y) { return 0; }), isSetUpInit {false}
  {}

template <typename T>
  Init_2D<T>::Init_2D(InitFunction initFunc)
  : initFunc {initFunc}, isSetUpInit {false}
  {}


template <typename T>
  inline void 
  Init_2D<T>::SetUpInit(Heat_2D<T> & obj)
  {
    T x{0}, y{0};
    for (size_type i = 1; i < obj.in.topology().__local_shape[0]-1; ++i)
    {
      for (size_type j = 1; j < obj.in.topology().__local_shape[1]-1; ++j)
      {
        x = (i+obj.in.topology().starts[0]-1) * obj.dxs[0];
        y = (j+obj.in.topology().starts[1]-1) * obj.dxs[1];

        obj.in(i,j) = initFunc(x,y);
        obj.out(i,j) = initFunc(x,y);
      }
    }

    isSetUpInit = true;
  }


} // namespace InitialConditions





} // namespace pde
} // namespace final_project







#endif // define FINAL_PROJECT_INITIALIZATIONS_C_HPP