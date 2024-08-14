///
/// @file InitializationsC.hpp
/// @brief Objects of Initialization Conditions of PDEs
///
/// @author LI Yihai
/// @version 6.0
///
#ifndef FINAL_PROJECT_INITIALIZATIONS_C_HPP
#define FINAL_PROJECT_INITIALIZATIONS_C_HPP

#pragma once 
#include <functional>

#include <pde/detials/Heat_2D.hpp>
#include <pde/detials/Heat_3D.hpp>


namespace final_project { namespace pde {


namespace InitialConditions {


/// @brief An object of Initial Condition in 2D space.
/// @tparam T Value type
template <typename T>
  class Init_2D
  {
    using InitFunction = std::function<T(T, T)>;

    public:
    Init_2D();
    Init_2D(InitFunction & );

    private:
    InitFunction initFunc;
    void SetUpInit(Heat_2D<T> &);
    Bool isSetUpInit;

  
    friend Heat_2D<T>;
  }; // class Init_2D


template <typename T>
  class Init_3D
  {
    using InitFunction = std::function<T(T, T, T)>;

    public:
    Init_3D();
    Init_3D(InitFunction &);

    private:
    InitFunction initFunc;
    void SetUpInit(Heat_3D<T> &);
    Bool isSetUpInit;


    friend Heat_3D<T>;
  }; // class Init_3D


} // namespace InitialConditions


} // namespace pde
} // namespace final_project





///
///
/// --------------------------- Inline Function Definitions  ---------------------------  ///
///
///



namespace final_project { namespace pde {


namespace InitialConditions {


/// @brief The empty Initial Condition (=0)
/// @tparam T  Value type
template <typename T>
  Init_2D<T>::Init_2D()
: initFunc([](T x, T y) { return 0; }), isSetUpInit {false}
  {}


/// @brief Use external function as initial condition.
/// @tparam T  Value type
/// @param initFunc 
template <typename T>
  Init_2D<T>::Init_2D(InitFunction & initFunc)
  : initFunc {initFunc}, isSetUpInit {false}
  {}

/// @brief Apply function to object.
/// @tparam T Value type
/// @param obj Reference of object Heat_2D<T>
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



/// @brief The empty Initial Condition (=0)
/// @tparam T  Value type
template <typename T>
  Init_3D<T>::Init_3D()
: initFunc([](T x, T y, T z) { return 0; }), isSetUpInit {false}
  {}


/// @brief Use external function as initial condition.
/// @tparam T  Value type
/// @param initFunc 
template <typename T>
  Init_3D<T>::Init_3D(InitFunction & initFunc)
  : initFunc {initFunc}, isSetUpInit {false}
  {}



/// @brief Apply function to object.
/// @tparam T Value type
/// @param obj Reference of object Heat_3D<T>
template <typename T>
  inline void 
  Init_3D<T>::SetUpInit(Heat_3D<T> & obj)
  {
    T x {0}, y {0}, z {0};
    for (size_type i = 1; i < obj.in.topology().__local_shape[0]-1; ++i)
    {
      for (size_type j = 1; j < obj.in.topology().__local_shape[1]-1; ++j)
      {
        for (size_type k = 1; k < obj.in.topology().__local_shape[2]-1; ++k)
        {
          x = (i + obj.in.topology().starts[0]-1) * obj.dxs[0];
          y = (j + obj.in.topology().starts[1]-1) * obj.dxs[1];
          y = (k + obj.in.topology().starts[2]-1) * obj.dxs[2];

          obj.in(i,j,k)  = initFunc(x,y,k);
          obj.out(i,j,k) = initFunc(x,y,k);
        }
      }
    }
  }












} // namespace InitialConditions








} // namespace pde
} // namespace final_project







#endif // define FINAL_PROJECT_INITIALIZATIONS_C_HPP