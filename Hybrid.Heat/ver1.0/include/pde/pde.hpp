///
/// @file pde/pde.hpp
/// @brief This file gives the prototypes of PDEs object and the
///         routines of namespace layouts inside of pde namespace.
///         
/// @see BoundaryConditions
/// @see InitialConditions
///
#ifndef FINAL_PROJECT_PDE_PDE_HPP
#define FINAL_PROJECT_PDE_PDE_HPP



#pragma once
#include <types.hpp>

#include <pde/Heat.hpp>
// #include <pde/Naiver_Stokes.hpp>

#include <pde/boundaryconditions/DirchletBC.hpp>
#include <pde/boundaryconditions/NeumannBC.hpp>



namespace final_project { 
    

/// @namespace pde
///   the namespace provides the solver of Partial Differential Equations.
///           
namespace pde {

/// @brief The basic abstract class of heat equation.
/// @tparam T The value type.
/// @tparam NumD The number of dimension.
template <typename T, size_type NumD>
  class Heat_Base;



template <typename T, size_type NumD>
  class Naiver_Stokes_Base;


/// @namespace BoundaryConditions
///   Contains the object that provides numbers if boundary condition types.
///   Operate to the defined @class array_Cart.
namespace BoundaryConditions {

template <typename T,  size_type NumD>
  class DirchletBC;

template <typename T,  size_type NumD>
  class NeumannBC;

  
} // namespace BoundaryConditions


/// @namespace InitialConditions
///   Give the initialization routines to the defined @class array_Cart.
namespace InitialConditions {

template <typename T, size_type NumD>
  class Init;
  
} // namespace InitialConditions


} // namespace pde
} // namespace final_project











#endif // end of define FINAL_PROJECT_PDE_PDE_HPP