///
/// @file   types.hpp
/// @brief  This file defines those general datatypes needed 
///         in this project
/// @author LI Yihai
///

#ifndef FINAL_PROJECT_TYPES_HPP_LIYIHAI
#define FINAL_PROJECT_TYPES_HPP_LIYIHAI


#include <cstdint>  // Standard integer types
#include <string>   // Include 'std::string' type
#include <iostream>

namespace final_project 
{

  // Type Aliases
  using Byte    = uint8_t;
  using Word    = uint16_t;
  using Dworld  = uint32_t;
  using Qworld  = uint64_t;

  // Common datatypes
  using Integer         = int;
  using UnsignedInteger = unsigned int;
  using Float           = float;
  using Double          = double;
  using Char            = char;
  using Bool            = bool;

  using String          = std::string;


  typedef Dworld    size_type;
  typedef Qworld    super_size_type;


} // end of namespace final_project

#endif // end of define FINAL_PROJECT_TYPES_HPP_LIYIHAI