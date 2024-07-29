#ifndef FINAL_PROJECT_HEAT_EQUATION_HPP
#define FINAL_PROJECT_HEAT_EQUATION_HPP



#pragma once
#include <types.hpp>


namespace final_project { namespace pde {


template <typename T, size_type NumD>
  class Heat;

template <typename T, size_type NumD>
  class Naiver_Stokes;


template <typename T, size_type NumD>
  class BoundaryConditions;

template <typename T>
  class InitialConditions;


// template <typename T>
//   class BoundaryConditions<T, 1> {

//   };








} // namespace pde
} // namespace final_project









#endif // end of define FINAL_PROJECT_HEAT_EQUATION_HPP