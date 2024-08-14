///
/// @file BoundaryConditions_3D.hpp
/// @brief Objects of Boundary Conditions of PDE. 
///         Includes Dirichlet and Von Neumann Boundary Conditions.
///
/// @author LI Yihai
/// @version 6.0
///

#ifndef FINAL_PROJECT_BOUNDARY_CONDITIONS_3D_HPP
#define FINAL_PROJECT_BOUNDARY_CONDITIONS_3D_HPP

#pragma once 
#include <functional>

#include <pde/detials/Heat_3D.hpp>

namespace final_project { namespace pde
{
  
template <typename T>
  class BoundaryConditions_3D
  {
    using BCFunction = std::function<T(T, T, T, T)>;

    public:
    BoundaryConditions_3D();                                          // An empty BC object.
    BoundaryConditions_3D(Bool, Bool, Bool, Bool, Bool, Bool, Bool);  // Setup the Boundary Condition Types.
    void SetBC(Heat_3D<T> &, 
      BCFunction, BCFunction, 
      BCFunction, BCFunction, 
      BCFunction, BCFunction);
    
    private:
    Bool isSetUpBC;
    std::array<Bool,        6> isDirichletBC;
    std::array<Bool,        6> isNeumann;
    std::array<BCFunction,  6> BCFunc;

    void SetBCDim000(Heat_3D<T> &, const T &);
    void SetBCDim001(Heat_3D<T> &, const T &);

    void SetBCDim010(Heat_3D<T> &, const T &);
    void SetBCDim011(Heat_3D<T> &, const T &);

    void SetBCDim100(Heat_3D<T> &, const T &);
    void SetBCDim101(Heat_3D<T> &, const T &);

    void UpdateBC(Heat_3D<T> &, const T);

    friend Heat_3D<T>;
  }; // class BoundaryConditions_3D


} // namespace pde
} // namespace final_project




#endif // end define FINAL_PROJECT_BOUNDARY_CONDITIONS_3D_HPP
