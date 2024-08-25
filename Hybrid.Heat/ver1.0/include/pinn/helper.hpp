#ifndef FINAL_PROJECT_PINN_HELPER_HPP_LIYIHAI
#define FINAL_PROJECT_PINN_HELPER_HPP_LIYIHAI

#pragma once
#include <types.hpp>
#include <iostream>

namespace final_project
{

  namespace PINN {
    enum class Dimension {
      PINN_2D,
      PINN_3D,
      UNKNOWN
    };
  
    Dimension getDimensionfromString(const String &);
    std::ostream & operator<<(std::ostream &, Dimension);

    void helper_message();
  }
}




#endif // end define FINAL_PROJECT_PINN_HELPER_HPP_LIYIHAI