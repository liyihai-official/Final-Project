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
    BoundaryConditions_2D();
    BoundaryConditions_2D(Bool, Bool, Bool, Bool);

    void SetBC(Heat_2D<T> &, T, T, T, T);
    void SetBC(Heat_2D<T> &, BCFunction, BCFunction, BCFunction, BCFunction);

    private:
    Bool isSetUpBC;
    std::array<Bool, 4> isDirichletBC;
    std::array<Bool, 4> isNeumann;
    std::array<T, 4>    BCValue;
    std::array<BCFunction, 4> BCFunc; 

    void SetBCinDim00(Heat_2D<T> &, T &);
    void SetBCinDim01(Heat_2D<T> &, T &);
    void SetBCinDim10(Heat_2D<T> &, T &);
    void SetBCinDim11(Heat_2D<T> &, T &);

    void SetBCinDim00(Heat_2D<T> &, BCFunction &);
    void SetBCinDim01(Heat_2D<T> &, BCFunction &);
    void SetBCinDim10(Heat_2D<T> &, BCFunction &);
    void SetBCinDim11(Heat_2D<T> &, BCFunction &);

    
    void UpdateBC(Heat_2D<T> &);

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
  BoundaryConditions_2D<T>::SetBC(Heat_2D<T> & obj, 
    BCFunction Funcdim00, BCFunction Funcdim01,   // is Neumann in Dimension 0, as Function g(x,y,t)
    BCFunction Funcdim10, BCFunction Funcdim11)   // is Neumann in Dimension 1, as Function g(x,y,t)
  {

    BCFunc = {Funcdim00, Funcdim01, Funcdim10, Funcdim11};

    SetBCinDim00(obj, BCFunc[0]);
    // SetBCinDim01(obj, BCFunc[1]);
    // SetBCinDim10(obj, BCFunc[2]);
    // SetBCinDim11(obj, BCFunc[3]);

    isSetUpBC = true;

    #ifndef NDEBUG
    std::cout << "Boundary Conditions are setup (rank " << obj.in.topology().rank << "): " 
              << " as function."
              << std::endl;
    #endif
  } 


template <typename T>
  inline void 
  BoundaryConditions_2D<T>::SetBC(Heat_2D<T> &obj, 
    T Vdim00, T Vdim01,   // in Dimension 0, as Constant, value or derivative
    T Vdim10, T Vdim11)   // in Dimension 1, as Constant, value or derivative
  {
    BCValue = {Vdim00, Vdim01, Vdim10, Vdim11};

    SetBCinDim00(obj, BCValue[0]);
    SetBCinDim01(obj, BCValue[1]);
    SetBCinDim10(obj, BCValue[2]);
    SetBCinDim11(obj, BCValue[3]);

    isSetUpBC = true;

  #ifndef NDEBUG
  std::cout << "Boundary Conditions are setup" 
            << " (rank " << obj.in.topology().rank << "): " 
            << " as constant value."
            << std::endl;
  #endif
  }

template <typename T>
  inline void
  BoundaryConditions_2D<T>::UpdateBC(Heat_2D<T> &obj)
  {
    FINAL_PROJECT_MPI_ASSERT_GLOBAL(isSetUpBC);

    SetBCinDim00(obj, BCValue[0]);
    SetBCinDim01(obj, BCValue[1]);
    SetBCinDim10(obj, BCValue[2]);
    SetBCinDim11(obj, BCValue[3]);
  }

template <typename T>
  inline void
  BoundaryConditions_2D<T>::SetBCinDim00(Heat_2D<T> & obj, T & Value)
  {
    auto shape_cpy  { obj.in.topology().__local_shape};
    auto starts_cpy {obj.in.topology().starts};

    T value {isDirichletBC[0] ? BCValue[0] : 0};

    if (starts_cpy[0] == 1)
    {
      size_type x = 0;
      for (size_type y = 1; y < shape_cpy[1] - 1; ++y)
      {
        if (isDirichletBC[0] && !isSetUpBC)
        {
          obj.in(x,y)  = value;
          obj.out(x,y) = value;
        } else if (isNeumann[0] && isSetUpBC)
        {
          obj.in(x,y)  += BCValue[0]*obj.dt;
          obj.out(x,y) += BCValue[0]*obj.dt;
        }
      }
    }
  }

template <typename T>
  inline void 
  BoundaryConditions_2D<T>::SetBCinDim01(Heat_2D<T> & obj, T & Value)
  {
    auto shape_cpy  { obj.in.topology().__local_shape};
    auto glob_cpy   { obj.in.topology().__global_shape};
    auto ends_cpy {obj.in.topology().ends};
    
    T value {isDirichletBC[1] ? BCValue[1] : 0};

    if (ends_cpy[0] == glob_cpy[0] - 2)
    {
      size_type x = shape_cpy[0]-1;
      for (size_type y = 1; y < shape_cpy[1] - 1; ++y)
      { 
        if (isDirichletBC[1] && !isSetUpBC)
        {
          obj.in(x,y)  = value;
          obj.out(x,y) = value;
        } else if (isNeumann[1] && isSetUpBC)
        {
          obj.in(x,y)  += BCValue[1]*obj.dt;
          obj.out(x,y) += BCValue[1]*obj.dt;
        }
      }
    }
  }

template <typename T>
  inline
  void
  BoundaryConditions_2D<T>::SetBCinDim10(Heat_2D<T> & obj, T & Value)
  {
    auto shape_cpy  {obj.in.topology().__local_shape};
    auto starts_cpy {obj.in.topology().starts};

    T value {isDirichletBC[2] ? BCValue[2] : 0};
    
    if (starts_cpy[1] == 1)
    {
      size_type y = 0;
      for (size_type x = 1; x < shape_cpy[0] - 1; ++x)
      { 
        if (isDirichletBC[2] && !isSetUpBC)
        {
          obj.in(x,y)  = value;
          obj.out(x,y) = value;
        } else if (isNeumann[2] && isSetUpBC)
        {
          obj.in(x,y)  += BCValue[2]*obj.dt;
          obj.out(x,y) += BCValue[2]*obj.dt;
        }
      }
    }
  }

template <typename T>
  inline 
  void 
  BoundaryConditions_2D<T>::SetBCinDim11(Heat_2D<T> & obj, T & Value)
  {
    auto shape_cpy  { obj.in.topology().__local_shape};
    auto glob_cpy   { obj.in.topology().__global_shape};
    auto ends_cpy {obj.in.topology().ends};
    
    T value {isDirichletBC[3] ? BCValue[3] : 0};

    if (ends_cpy[1] == glob_cpy[1] - 2 )
    {
      size_type y = shape_cpy[1]-1;
      for (size_type x = 1; x < shape_cpy[0] - 1; ++x)
      { 
        if (isDirichletBC[3] && !isSetUpBC)
        {
          obj.in(x,y)  = value;
          obj.out(x,y) = value;
        } else if (isNeumann[3] && isSetUpBC)
        {
          obj.in(x,y)  += BCValue[3]*obj.dt;
          obj.out(x,y) += BCValue[3]*obj.dt;
        }
      }
    }
  }
  
//////////////////////////////////////////////////////////////////////////////////////////
// Requires Time (iteration).
template <typename T>
  inline void
  BoundaryConditions_2D<T>::SetBCinDim00(Heat_2D<T> & obj, BCFunction & func)
  {
    auto shape_cpy  { obj.in.topology().__local_shape};
    auto starts_cpy {obj.in.topology().starts};

    // T value {isDirichletBC[0] ? BCValue[0] : 0};


    // T x {0}, y {0};
    // if (starts_cpy[0] == 1)
    // {
    //   size_type i = 0;
    //   x = (i + obj.in.topology().starts[0] - 1) * obj.dxs[0];
    //   for (size_type j = 1; j < shape_cpy[1] - 1; ++j)
    //   {
    //     y = (j + obj.in.topology().starts[1] - 1) * obj.dxs[1];

    //     if (isDirichletBC[0] && !isSetUpBC)
    //     {
    //       obj.in(i,j) = func(x,y,0);
    //       obj.out(i,j) = func(x,y,0);
    //     } else if (isNeumann[0] && isSetUpBC)
    //     {
    //       obj.in(i,j) = func(x,y)
    //     }

    //   }
    //   for (size_type y = 1; y < shape_cpy[1] - 1; ++y)
    //   {
    //     if (isDirichletBC[0] && !isSetUpBC)
    //     {
    //       obj.in(x,y)  = value;
    //       obj.out(x,y) = value;
    //     } else if (isNeumann[0] && isSetUpBC)
    //     {
    //       obj.in(x,y)  += BCValue[0]*obj.dt;
    //       obj.out(x,y) += BCValue[0]*obj.dt;
    //     }
    //   }
    // }
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