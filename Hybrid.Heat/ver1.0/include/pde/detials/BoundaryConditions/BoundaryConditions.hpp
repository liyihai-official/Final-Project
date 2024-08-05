#ifndef FINAL_PROJECT_DIRCHLETBC_HPP
#define FINAL_PROJECT_DIRCHLETBC_HPP

#pragma once 
#include <functional>

#include <pde/detials/Heat_2D.hpp>


namespace final_project { namespace pde {


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

}}




///
///
/// Definitions of inline Functions
///
///


namespace final_project { namespace pde {
    


template <typename T>
  inline
  BoundaryConditions_2D<T>::BoundaryConditions_2D()
: isSetUpBC {false} {}


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
    std::cout << "Boundary Conditions are setup (rank " << obj.in.topology().rank << "): " 
              << " as function."
              << std::endl;
    #endif

  }


template <typename T>
  inline void 
  BoundaryConditions_2D<T>::UpdateBC(
    Heat_2D<T> & obj, 
    const T time)
  {
    FINAL_PROJECT_MPI_ASSERT_GLOBAL(isSetUpBC);

    SetBCinDim00(obj, time);
    SetBCinDim01(obj, time);
    SetBCinDim10(obj, time);
    SetBCinDim11(obj, time);
  }




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




#endif // end define FINAL_PROJECT_DIRCHLETBC_HPP












    // {
    //   auto shape_cpy  { obj.in.topology().__local_shape};
    //   auto glob_cpy   { obj.in.topology().__global_shape};

    //   auto starts_cpy {obj.in.topology().starts};
    //   auto ends_cpy {obj.in.topology().ends};

    //   if (starts_cpy[0] == 1)
    //   {
    //     size_type x = 0;
    //     for (size_type y = 1; y < shape_cpy[1] - 1; ++y){ obj.in(x,y) = v00; obj.out(x,y) = v00;}
    //   }

    //   if (ends_cpy[0] == glob_cpy[0] - 2)
    //   {
    //     size_type x = shape_cpy[0]-1;
    //     for (size_type y = 1; y < shape_cpy[1] - 1; ++y){ obj.in(x,y) = v01; obj.out(x,y) = v01;}
    //   }

    //   if (starts_cpy[1] == 1 )
    //   {
    //     size_type y = 0;
    //     for (size_type x = 1; x < shape_cpy[0] - 1; ++x){ obj.in(x,y) = v10; obj.out(x,y) = v10;}
    //   }

    //   if (ends_cpy[1] == glob_cpy[1] - 2)
    //   {
    //     size_type y = shape_cpy[1]-1;
    //     for (size_type x = 1; x < shape_cpy[0] - 1; ++x) {obj.in(x,y) = v11; obj.out(x,y) = v11;}
    //   }
    // }