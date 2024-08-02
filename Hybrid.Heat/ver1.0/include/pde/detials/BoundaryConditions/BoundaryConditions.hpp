#ifndef FINAL_PROJECT_DIRCHLETBC_HPP
#define FINAL_PROJECT_DIRCHLETBC_HPP

#pragma once 
#include <pde/detials/Heat_2D.hpp>


namespace final_project { namespace pde {


template <typename T>
  class BoundaryConditions_2D
  {
    public:
    BoundaryConditions_2D();

    BoundaryConditions_2D(Bool, Bool, Bool, Bool);
    
    BoundaryConditions_2D(Heat_2D<T> & obj, T v00, T v01, T v10, T v11)
    {
      auto shape_cpy  { obj.in.topology().__local_shape};
      auto glob_cpy   { obj.in.topology().__global_shape};

      auto starts_cpy {obj.in.topology().starts};
      auto ends_cpy {obj.in.topology().ends};

      if (starts_cpy[0] == 1)
      {
        size_type x = 0;
        for (size_type y = 1; y < shape_cpy[1] - 1; ++y){ obj.in(x,y) = v00; obj.out(x,y) = v00;}
      }

      if (ends_cpy[0] == glob_cpy[0] - 2)
      {
        size_type x = shape_cpy[0]-1;
        for (size_type y = 1; y < shape_cpy[1] - 1; ++y){ obj.in(x,y) = v01; obj.out(x,y) = v01;}
      }

      if (starts_cpy[1] == 1 )
      {
        size_type y = 0;
        for (size_type x = 1; x < shape_cpy[0] - 1; ++x){ obj.in(x,y) = v10; obj.out(x,y) = v10;}
      }

      if (ends_cpy[1] == glob_cpy[1] - 2)
      {
        size_type y = shape_cpy[1]-1;
        for (size_type x = 1; x < shape_cpy[0] - 1; ++x) {obj.in(x,y) = v11; obj.out(x,y) = v11;}
      }
    }

    private:
    std::array<T, 4>    BCValue;
    std::array<Bool, 4> isDirchletBC;
    std::array<Bool, 4> isNeumann;

    void SetBCinDim00(Heat_2D<T> &);
    void SetBCinDim01(Heat_2D<T> &);
    void SetBCinDim10(Heat_2D<T> &);
    void SetBCinDim11(Heat_2D<T> &);

    public:
    void SetBC(Heat_2D<T> &, T, T, T, T);
  };


}}




///
///
/// Definitions of inline Functions
///
///


namespace final_project { namespace pde {
    


// template <typename T>
//   inline
//   BoundaryConditions_2D<T>::BoundaryConditions_2D()
// : DirchletBCValue ({0, 0, 0, 0})
// , isDirchletBC ({true, true, true, true})
// , isNeumann ({false, false, false, false})
//   {}


// template <typename T>
//   inline
//   BoundaryConditions_2D<T>::BoundaryConditions_2D(
//     Bool isDirchDim00, Bool isDirchDim01,   // is Dirchlet in Dimension 0
//     Bool isDirchDim10, Bool isDirchDim11)   // is Dirchlet in Dimension 1
//   {
//     isDirchletBC = {isDirchDim00, isDirchDim01, isDirchDim10, isDirchDim11};
//     for (Integer i = 0; i < 4; ++i)
//     {
//       if (!isDirchletBC[i]) isNeumann[i] = true;
//     }
//   }


// template <typename T>
//   inline
//   void
//   BoundaryConditions_2D<T>::SetBCinDim00(Heat_2D<T> & obj)
//   {
//     auto shape_cpy  { obj.in.topology().__local_shape};
//     auto starts_cpy {obj.in.topology().starts};

//     if (starts_cpy[0] == 1 && isDirchletBC[0])
//     {
//       size_type x = 0;
//       for (size_type y = 1; y < shape_cpy[1] - 1; ++y)
//       { 
//         obj.in(x,y)  = BCValue[0];
//         obj.out(x,y) = BCValue[0];
//       }
//     }
//   }

// template <typename T>
//   inline 
//   void 
//   BoundaryConditions_2D<T>::SetBCinDim01(Heat_2D<T> & obj)
//   {
//     auto shape_cpy  { obj.in.topology().__local_shape};
//     auto glob_cpy   { obj.in.topology().__global_shape};
//     auto ends_cpy {obj.in.topology().ends};
    
//     if (ends_cpy[0] == glob_cpy[0] - 2  && isDirchletBC[1])
//     {
//       size_type x = shape_cpy[0]-1;
//       for (size_type y = 1; y < shape_cpy[1] - 1; ++y)
//       { 
//         obj.in(x,y)  = BCValue[1];
//         obj.out(x,y) = BCValue[1];
//       }
//     }
//   }

// template <typename T>
//   inline
//   void
//   BoundaryConditions_2D<T>::SetBCinDim10(Heat_2D<T> & obj)
//   {
//     auto shape_cpy  {obj.in.topology().__local_shape};
//     auto starts_cpy {obj.in.topology().starts};

//     if (starts_cpy[1] == 1 && isDirchletBC[2])
//     {
//       size_type y = 0;
//       for (size_type x = 1; x < shape_cpy[0] - 1; ++x)
//       { 
//         obj.in(x,y)  = BCValue[2];
//         obj.out(x,y) = BCValue[2];
//       }
//     }
//   }

// template <typename T>
//   inline 
//   void 
//   BoundaryConditions_2D<T>::SetBCinDim11(Heat_2D<T> & obj)
//   {
//     auto shape_cpy  { obj.in.topology().__local_shape};
//     auto glob_cpy   { obj.in.topology().__global_shape};
//     auto ends_cpy {obj.in.topology().ends};
    
//     if (ends_cpy[1] == glob_cpy[1] - 2  && isDirchletBC[3])
//     {
//       size_type y = shape_cpy[1]-1;
//       for (size_type x = 1; x < shape_cpy[0] - 1; ++x)
//       { 
//         obj.in(x,y)  = BCValue[3];
//         obj.out(x,y) = BCValue[3];
//       }
//     }
//   }


// template <typename T>
//   inline 
//   void 
//   BoundaryConditions_2D<T>::SetBC(Heat_2D<T> &obj, T Vdim00, T Vdim01, T Vdim10, T Vdim11)
//   {
//     BCValue = {Vdim00, Vdim01, Vdim10, Vdim11};

//     SetBCinDim00
//   }
  


} // namespace pde
} // namespace final_project




#endif // end define FINAL_PROJECT_DIRCHLETBC_HPP