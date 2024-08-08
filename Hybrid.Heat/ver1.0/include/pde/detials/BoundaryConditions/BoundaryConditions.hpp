///
/// @file BoundaryConditions.hpp
/// @brief Objects of Boundary Conditions of PDE. 
///         Includes Dirichlet and Von Neumann Boundary Conditions.
///
/// @author LI Yihai
/// @version 6.0
///

#ifndef FINAL_PROJECT_BOUNDARY_CONDITIONS_HPP
#define FINAL_PROJECT_BOUNDARY_CONDITIONS_HPP

#pragma once 
#include <functional>

#include <pde/detials/Heat_2D.hpp>


namespace final_project { namespace pde {


/// @brief Boundary Conditions in 2 Dimension space, g(x,y,t) on the Boundaries.
/// @tparam T Value type.
template <typename T>
  class BoundaryConditions_2D
  {
    using BCFunction = std::function<T(T, T, T)>;

    public:
    BoundaryConditions_2D();                        // An empty BC object.
    BoundaryConditions_2D(Bool, Bool, Bool, Bool);  // Setup the Boundary Condition Types.
    void SetBC(Heat_2D<T> &, BCFunction, BCFunction, BCFunction, BCFunction);     // Setup the function of Boundary Conditions

    private:
    Bool isSetUpBC;
    std::array<Bool, 4> isDirichletBC;
    std::array<Bool, 4> isNeumann;
    std::array<BCFunction, 4> BCFunc; 

    void SetBCinDim00(Heat_2D<T> &, const T &);
    void SetBCinDim01(Heat_2D<T> &, const T &);
    void SetBCinDim10(Heat_2D<T> &, const T &);
    void SetBCinDim11(Heat_2D<T> &, const T &);

    // void UpdateBC(Heat_2D<T> &);
    void UpdateBC(Heat_2D<T> &, const T);

    friend Heat_2D<T>;
  };

} // namespace pde
} // namespace final_project







///
///
/// --------------------------- Inline Function Definitions  ---------------------------  ///
///
///




namespace final_project { namespace pde {
    

/// @brief The empty constructor
/// @tparam T Value type
template <typename T>
  inline
  BoundaryConditions_2D<T>::BoundaryConditions_2D()
: isSetUpBC {false} {}


/// @brief Specifying the Type of boundary conditions in each edge.
/// @tparam T Value type
/// @param isDirichDim00 true if it's Dirichlet in Dimension 0, source site.
/// @param isDirichDim01 true if it's Dirichlet in Dimension 0, dest site.
/// @param isDirichDim10 true if it's Dirichlet in Dimension 1, source site.
/// @param isDirichDim11 true if it's Dirichlet in Dimension 1, dest site.
template <typename T>
  inline
  BoundaryConditions_2D<T>::BoundaryConditions_2D(
    Bool isDirichDim00, Bool isDirichDim01,   // is Dirichlet in Dimension 0, as Constant Value 
    Bool isDirichDim10, Bool isDirichDim11)   // is Dirichlet in Dimension 1, as Constant Value 
  {
    isDirichletBC = {isDirichDim00, isDirichDim01, isDirichDim10, isDirichDim11};
    for (Integer i = 0; i < 4; ++i)
    {
  if (!isDirichletBC[i]) isNeumann[i] = true;
    }
  }


/// @brief Set Up the Boundary Conditions with std::function
/// @tparam T Value type
/// @param obj The reference of the object Heat_2D<T>.
/// @param FuncDim00 The function in Dimension 0, source site.
/// @param FuncDim01 The function in Dimension 0, dest site.
/// @param FuncDim10 The function in Dimension 1, source site.
/// @param FuncDim11 The function in Dimension 1, dest site.
template <typename T>
  inline void 
  BoundaryConditions_2D<T>::SetBC(
    Heat_2D<T> & obj, 
    BCFunction FuncDim00, BCFunction FuncDim01,   // is Neumann or Dirichlet in Dimension 0, as Function g(x,y,t)
    BCFunction FuncDim10, BCFunction FuncDim11)   // is Neumann or Dirichlet in Dimension 1, as Function g(x,y,t)
  {
    BCFunc = {FuncDim00, FuncDim01, FuncDim10, FuncDim11};

    SetBCinDim00(obj, 0);
    SetBCinDim01(obj, 0);
    SetBCinDim10(obj, 0);
    SetBCinDim11(obj, 0);
    
    isSetUpBC = true;

#ifndef NDEBUG
std::cout << "Boundary Conditions are setup (rank " 
  << obj.in.topology().rank << "): " 
  << " as function."
  << std::endl;
#endif

  }


/// @brief Update the Boundary Conditions During Evolving Iterations.
/// @tparam T Value type
/// @param obj The reference of the object Heat_2D<T>.
/// @param time Time of Evolving Processes.
template <typename T>
  inline void 
  BoundaryConditions_2D<T>::UpdateBC(
    Heat_2D<T> & obj, 
    const T time)
  {
FINAL_PROJECT_MPI_ASSERT_GLOBAL(isSetUpBC); // The BCs have to be setup.

    SetBCinDim00(obj, time);
    SetBCinDim01(obj, time);
    SetBCinDim10(obj, time);
    SetBCinDim11(obj, time);
  }



/// @brief Setup the Boundary Condition in Dimension 0, on the source site (0). 
/// @tparam T Value type
/// @param obj The reference of the object Heat_2D<T>.
/// @param time Time of Evolving Processes.
template <typename T>
  inline void 
  BoundaryConditions_2D<T>::SetBCinDim00(
    Heat_2D<T> & obj, 
    const T & time)
  {
    auto shape_cpy  { obj.in.topology().__local_shape };
    auto starts_cpy { obj.in.topology().starts };

    T x {0}, y {0};

    if (starts_cpy[0] == 1)
    {
      size_type i {0};
      for (size_type j = 1; j < shape_cpy[1]-1; ++j)
      {
x = ( i + starts_cpy[0] - 1) * obj.dxs[0];
y = ( j + starts_cpy[1] - 1) * obj.dxs[1];

if (isDirichletBC[0])
{
  obj.in(i,j)  = BCFunc[0](x,y,time);
  obj.out(i,j) = BCFunc[0](x,y,time);
}
else if (isNeumann[0])
{
  obj.in(i,j)  += BCFunc[0](x,y,time)*obj.dt;
  obj.out(i,j) += BCFunc[0](x,y,time)*obj.dt;
}
      }
    }    
  }

/// @brief Setup the Boundary Condition in Dimension 0, on the dest site (1).
/// @tparam T Value type
/// @param obj The reference of the object Heat_2D<T>.
/// @param time Time of Evolving Processes.
template <typename T>
  inline void 
  BoundaryConditions_2D<T>::SetBCinDim01(
    Heat_2D<T> & obj,
    const T & time)
  {
    auto shape_cpy  { obj.in.topology().__local_shape};
    auto glob_cpy   { obj.in.topology().__global_shape};

    auto starts_cpy { obj.in.topology().starts };
    auto ends_cpy   { obj.in.topology().ends};

    T x {0}, y {0};

    if (ends_cpy[0] == glob_cpy[0] - 2)
    { 
      size_type i {shape_cpy[0]-1};
      for (size_type j = 1; j < shape_cpy[1]-1; ++j)
      {
x = ( i + starts_cpy[0] - 1) * obj.dxs[0];
y = ( j + starts_cpy[1] - 1) * obj.dxs[1];

if (isDirichletBC[1])
{
  obj.in(i,j)  = BCFunc[1](x,y,time);
  obj.out(i,j) = BCFunc[1](x,y,time);
}
else if (isNeumann[1])
{
  obj.in(i,j)  += BCFunc[1](x,y,time)*obj.dt;
  obj.out(i,j) += BCFunc[1](x,y,time)*obj.dt;
}
      }
    }

  }





/// @brief Setup the Boundary Condition in Dimension 1, on the dest source (0).
/// @tparam T Value type
/// @param obj The reference of the object Heat_2D<T>.
/// @param time Time of Evolving Processes.
template <typename T>
  inline void 
  BoundaryConditions_2D<T>::SetBCinDim10(
    Heat_2D<T> & obj,
    const T & time)
  {
    auto shape_cpy  { obj.in.topology().__local_shape};
    auto glob_cpy   { obj.in.topology().__global_shape};

    auto starts_cpy { obj.in.topology().starts };
    // auto ends_cpy {obj.in.topology().ends};

    T x {0}, y {0};

    if (starts_cpy[1] == 1)
    {
      size_type j {0};
      for (size_type i = 1; i < shape_cpy[0]-1; ++i)
      {
x = ( i + starts_cpy[0] - 1) * obj.dxs[0];
y = ( j + starts_cpy[1] - 1) * obj.dxs[1];

if (isDirichletBC[2])
{
  obj.in(i,j)  = BCFunc[2](x,y,time);
  obj.out(i,j) = BCFunc[2](x,y,time);
}
else if (isNeumann[2])
{
  obj.in(i,j)  += BCFunc[2](x,y,time)*obj.dt;
  obj.out(i,j) += BCFunc[2](x,y,time)*obj.dt;
}
      }
    }
  }

/// @brief Setup the Boundary Condition in Dimension 1, on the dest site (1).
/// @tparam T Value type
/// @param obj The reference of the object Heat_2D<T>.
/// @param time Time of Evolving Processes.
template <typename T>
  inline void 
  BoundaryConditions_2D<T>::SetBCinDim11(
    Heat_2D<T> & obj,
    const T & time)
  {
    auto shape_cpy  { obj.in.topology().__local_shape};
    auto glob_cpy   { obj.in.topology().__global_shape};

    auto starts_cpy { obj.in.topology().starts };
    auto ends_cpy   {obj.in.topology().ends};

    T x {0}, y {0};

    if (ends_cpy[1] == glob_cpy[1] - 2 )
    {
      size_type j {shape_cpy[1]-1};
      for (size_type i = 1; i < shape_cpy[0]-1; ++i)
      {
x = ( i + starts_cpy[0] - 1) * obj.dxs[0];
y = ( j + starts_cpy[1] - 1) * obj.dxs[1];

if (isDirichletBC[3])
{
  obj.in(i,j)  = BCFunc[3](x,y,time);
  obj.out(i,j) = BCFunc[3](x,y,time);
}
else if (isNeumann[3])
{
  obj.in(i,j)  += BCFunc[3](x,y,time)*obj.dt;
  obj.out(i,j) += BCFunc[3](x,y,time)*obj.dt;
}
      }
    }
  }



} // namespace pde
} // namespace final_project


#endif // end define FINAL_PROJECT_BOUNDARY_CONDITIONS_HPP

