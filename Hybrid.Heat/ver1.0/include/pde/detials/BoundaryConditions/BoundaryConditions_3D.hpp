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
    BoundaryConditions_3D(Bool, Bool, Bool, Bool, Bool, Bool);  // Setup the Boundary Condition Types.
    void SetBC(Heat_3D<T> &, 
      BCFunction, BCFunction, 
      BCFunction, BCFunction, 
      BCFunction, BCFunction);
    
    private:
    Bool isSetUpBC;
    std::array<Bool,        6> isDirichletBC;
    std::array<Bool,        6> isNeumann;
    std::array<BCFunction,  6> BCFunc;

    void SetBCinDim00(Heat_3D<T> &, const T &);
    void SetBCinDim01(Heat_3D<T> &, const T &);

    void SetBCinDim10(Heat_3D<T> &, const T &);
    void SetBCinDim11(Heat_3D<T> &, const T &);

    void SetBCinDim20(Heat_3D<T> &, const T &);
    void SetBCinDim21(Heat_3D<T> &, const T &);

    void UpdateBC(Heat_3D<T> &, const T &);

    friend Heat_3D<T>;
  }; // class BoundaryConditions_3D


} // namespace pde
} // namespace final_project


///
///
/// --------------------------- Inline Function Definitions  ---------------------------  ///
///
///




namespace final_project { namespace pde {


template <typename T>
  inline BoundaryConditions_3D<T>::BoundaryConditions_3D()
: isSetUpBC {false} {}



/// @brief Specifying the Type of boundary conditions in each edge.
/// @tparam T Value type
/// @param isDirichDim00 true if it's Dirichlet in Dimension 0, source site.
/// @param isDirichDim01 true if it's Dirichlet in Dimension 0, dest site.
/// @param isDirichDim10 true if it's Dirichlet in Dimension 1, source site.
/// @param isDirichDim11 true if it's Dirichlet in Dimension 1, dest site.
/// @param isDirichDim20 true if it's Dirichlet in Dimension 2, source site.
/// @param isDirichDim21 true if it's Dirichlet in Dimension 2, dest site.
template <typename T>
  inline 
  BoundaryConditions_3D<T>::BoundaryConditions_3D(
    Bool isDirichDim00, Bool isDirichDim01,   // is Dirichlet in Dimension 0, as Constant Value 
    Bool isDirichDim10, Bool isDirichDim11,   // is Dirichlet in Dimension 1, as Constant Value 
    Bool isDirichDim20, Bool isDirichDim21)   // is Dirichlet in Dimension 2, as Constant Value 
  {
    isDirichletBC = {
      isDirichDim00, isDirichDim01,
      isDirichDim10, isDirichDim11,
      isDirichDim20, isDirichDim21
    };

    for (Integer i = 0; i < 6; ++i) { 
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
/// @param FuncDim20 The function in Dimension 2, source site.
/// @param FuncDim21 The function in Dimension 2, dest site.
template <typename T>
  inline void 
  BoundaryConditions_3D<T>::SetBC(
    Heat_3D<T> & obj,
    BCFunction FuncDim00, BCFunction FuncDim01,
    BCFunction FuncDim10, BCFunction FuncDim11,
    BCFunction FuncDim20, BCFunction FuncDim21)
  {
    BCFunc = {FuncDim00, FuncDim01, 
      FuncDim10, FuncDim11, 
      FuncDim20, FuncDim21};

    SetBCinDim00(obj, 0);
    SetBCinDim01(obj, 0);

    SetBCinDim10(obj, 0);
    SetBCinDim11(obj, 0);

    SetBCinDim20(obj, 0);
    SetBCinDim21(obj, 0);

    isSetUpBC = true;

#ifndef NDEBUG
std::cout << "Boundary Conditions are setup (rank " 
  << obj.in.topology().rank << "): " 
  << " as function."
  << std::endl;
#endif
  }


template <typename T>
  inline void 
  BoundaryConditions_3D<T>::UpdateBC(
    Heat_3D<T> & obj,
    const T & time)
  {
FINAL_PROJECT_MPI_ASSERT_GLOBAL(isSetUpBC); // The BCs have to be setup.

    SetBCinDim00(obj, time);
    SetBCinDim01(obj, time);

    SetBCinDim10(obj, time);
    SetBCinDim11(obj, time);

    SetBCinDim20(obj, time);
    SetBCinDim21(obj, time);
  }

template <typename T>
  inline void 
  BoundaryConditions_3D<T>::SetBCinDim00(
    Heat_3D<T> & obj, 
    const T & time)
  {
    auto shape_cpy { obj.in.topology().__local_shape };
    auto starts_cpy { obj.in.topology().starts };

    T x {0}, y {0}, z {0};

    if (starts_cpy[0] == 1)
    {
      size_type i {0};
      for (size_type j = 1; j < shape_cpy[1] - 1; ++j)
      {
        for (size_type k = 1; k < shape_cpy[2] - 1; ++k)
        {
x = ( i + starts_cpy[0] - 1) * obj.dxs[0];
y = ( j + starts_cpy[1] - 1) * obj.dxs[1];
z = ( k + starts_cpy[2] - 1) * obj.dxs[2];

if (isDirichletBC[0])
{
  obj.in(i,j,k)  = BCFunc[0](x,y,z,time);
  obj.out(i,j,k) = BCFunc[0](x,y,z,time);

} else if (isNeumann[0])
{
  obj.in(i,j,k)  += BCFunc[0](x,y,z,time) * obj.dt;
  obj.out(i,j,k) += BCFunc[0](x,y,z,time) * obj.dt;
}
        }
      }
    }
  }

template <typename T>
  inline void 
  BoundaryConditions_3D<T>::SetBCinDim01(
    Heat_3D<T> & obj, 
    const T & time)
  {
    auto shape_cpy  { obj.in.topology().__local_shape};
    auto glob_cpy   { obj.in.topology().__global_shape};

    auto starts_cpy { obj.in.topology().starts };
    auto ends_cpy   { obj.in.topology().ends};

    T x {0}, y {0}, z {0};

    if (ends_cpy[0] == glob_cpy[0] - 2)
    {
      size_type i { shape_cpy[0] - 1 };
      for (size_type j = 1; j < shape_cpy[1] - 1; ++j)
      {
        for (size_type k = 1; k < shape_cpy[2] - 1; ++k)
        {
x = ( i + starts_cpy[0] - 1) * obj.dxs[0];
y = ( j + starts_cpy[1] - 1) * obj.dxs[1];
z = ( k + starts_cpy[2] - 1) * obj.dxs[2];

if (isDirichletBC[1])
{
  obj.in(i,j,k)  = BCFunc[1](x,y,z,time);
  obj.out(i,j,k) = BCFunc[1](x,y,z,time);

} else if (isNeumann[1])
{
  obj.in(i,j,k)  += BCFunc[1](x,y,z,time) * obj.dt;
  obj.out(i,j,k) += BCFunc[1](x,y,z,time) * obj.dt;
}
        }
      }
    }
  }


template <typename T>
  inline void 
  BoundaryConditions_3D<T>::SetBCinDim10(
    Heat_3D<T> & obj, 
    const T & time)
  {
    auto shape_cpy { obj.in.topology().__local_shape };
    auto starts_cpy { obj.in.topology().starts };

    T x {0}, y {0}, z {0};

    if (starts_cpy[1] == 1)
    {
      size_type j {0};
      for (size_type i = 1; i < shape_cpy[0] - 1; ++i)
      {
        for (size_type k = 1; k < shape_cpy[2] - 1; ++k)
        {
x = ( i + starts_cpy[0] - 1) * obj.dxs[0];
y = ( j + starts_cpy[1] - 1) * obj.dxs[1];
z = ( k + starts_cpy[2] - 1) * obj.dxs[2];

if (isDirichletBC[2])
{
  obj.in(i,j,k)  = BCFunc[2](x,y,z,time);
  obj.out(i,j,k) = BCFunc[2](x,y,z,time);

} else if (isNeumann[2])
{
  obj.in(i,j,k)  += BCFunc[2](x,y,z,time) * obj.dt;
  obj.out(i,j,k) += BCFunc[2](x,y,z,time) * obj.dt;
}
        }
      }
    }
  }

template <typename T>
  inline void 
  BoundaryConditions_3D<T>::SetBCinDim11(
    Heat_3D<T> & obj, 
    const T & time)
  {
    auto shape_cpy  { obj.in.topology().__local_shape};
    auto glob_cpy   { obj.in.topology().__global_shape};

    auto starts_cpy { obj.in.topology().starts };
    auto ends_cpy   { obj.in.topology().ends};

    T x {0}, y {0}, z {0};

    if (ends_cpy[1] == glob_cpy[1] - 2)
    {
      size_type j { shape_cpy[1] - 1 };
      for (size_type i = 1; i < shape_cpy[0] - 1; ++i)
      {
        for (size_type k = 1; k < shape_cpy[2] - 1; ++k)
        {
x = ( i + starts_cpy[0] - 1) * obj.dxs[0];
y = ( j + starts_cpy[1] - 1) * obj.dxs[1];
z = ( k + starts_cpy[2] - 1) * obj.dxs[2];

if (isDirichletBC[3])
{
  obj.in(i,j,k)  = BCFunc[3](x,y,z,time);
  obj.out(i,j,k) = BCFunc[3](x,y,z,time);

} else if (isNeumann[3])
{
  obj.in(i,j,k)  += BCFunc[3](x,y,z,time) * obj.dt;
  obj.out(i,j,k) += BCFunc[3](x,y,z,time) * obj.dt;
}
        }
      }
    }
  }

template <typename T>
  inline void 
  BoundaryConditions_3D<T>::SetBCinDim20(
    Heat_3D<T> & obj, 
    const T & time)
  {
    auto shape_cpy { obj.in.topology().__local_shape };
    auto starts_cpy { obj.in.topology().starts };

    T x {0}, y {0}, z {0};

    if (starts_cpy[2] == 1)
    {
      size_type k {0};
      for (size_type i = 1; i < shape_cpy[0] - 1; ++i)
      {
        for (size_type j = 1; j < shape_cpy[1] - 1; ++j)
        {
x = ( i + starts_cpy[0] - 1) * obj.dxs[0];
y = ( j + starts_cpy[1] - 1) * obj.dxs[1];
z = ( k + starts_cpy[2] - 1) * obj.dxs[2];

if (isDirichletBC[4])
{
  obj.in(i,j,k)  = BCFunc[4](x,y,z,time);
  obj.out(i,j,k) = BCFunc[4](x,y,z,time);

} else if (isNeumann[4])
{
  obj.in(i,j,k)  += BCFunc[4](x,y,z,time) * obj.dt;
  obj.out(i,j,k) += BCFunc[4](x,y,z,time) * obj.dt;
}
        }
      }
    }
  }

template <typename T>
  inline void 
  BoundaryConditions_3D<T>::SetBCinDim21(
    Heat_3D<T> & obj, 
    const T & time)
  {
    auto shape_cpy  { obj.in.topology().__local_shape};
    auto glob_cpy   { obj.in.topology().__global_shape};

    auto starts_cpy { obj.in.topology().starts };
    auto ends_cpy   { obj.in.topology().ends};

    T x {0}, y {0}, z {0};

    if (ends_cpy[2] == glob_cpy[2] - 2)
    {
      size_type k { shape_cpy[2] - 1 };
      for (size_type i = 1; i < shape_cpy[0] - 1; ++i)
      {
        for (size_type j = 1; j < shape_cpy[1] - 1; ++j)
        {
x = ( i + starts_cpy[0] - 1) * obj.dxs[0];
y = ( j + starts_cpy[1] - 1) * obj.dxs[1];
z = ( k + starts_cpy[2] - 1) * obj.dxs[2];

if (isDirichletBC[5])
{
  obj.in(i,j,k)  = BCFunc[5](x,y,z,time);
  obj.out(i,j,k) = BCFunc[5](x,y,z,time);

} else if (isNeumann[5])
{
  obj.in(i,j,k)  += BCFunc[5](x,y,z,time) * obj.dt;
  obj.out(i,j,k) += BCFunc[5](x,y,z,time) * obj.dt;
}
        }
      }
    }
  }


} // namespace pde
} // namespace final_project

#endif // end define FINAL_PROJECT_BOUNDARY_CONDITIONS_3D_HPP
