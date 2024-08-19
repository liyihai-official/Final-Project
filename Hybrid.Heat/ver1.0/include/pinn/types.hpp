///
///
/// @file types.hpp
/// @brief Types for PINN, and Libtorch
/// 
///



#ifndef FINAL_PROJECT_PINN_TYPES_HPP
#define FINAL_PROJECT_PINN_TYPES_HPP


#pragma once
#include <types.hpp>
#include <torch/torch.h>

#if !defined (IN_SIZE_2D) || !defined(IN_SIZE_3D) || !defined (OUT_SIZE)
#define IN_SIZE_2D 2
#define IN_SIZE_3D 3
#define OUT_SIZE 1
#endif 

#if !defined(NX) || !defined (NY) || !defined(NZ)
#define NX 100+2
#define NY 100+2
#define NZ 50+2
#endif

#if !defined(DATASET_X_BOUNDARY) || !defined(DATASET_X_INTERNAL) || !defined(DATASET_Y_BOUNDARY)
#define DATASET_X_INTERNAL "X_internal.pt"
#define DATASET_X_BOUNDARY "X_boundary.pt"
#define DATASET_Y_BOUNDARY "Y_boundary.pt"
#endif


/// Using datatypes
using Integer = final_project::Integer;
using Char    = final_project::Char;
using Double  = final_project::Double;
using Float   = final_project::Float;
using String  = final_project::String;

using size_type = final_project::Dworld;

using maintype  = Float;

using BCFunction    = std::function<maintype(maintype, maintype)>;
using BCFunction_3d = std::function<maintype(maintype, maintype, maintype)>;


namespace final_project { namespace PINN {


typedef Float value_type;





} // namespace PINN
} // namespace final_project





#endif // end define FINAL_PROJECT_PINN_TYPES_HPP