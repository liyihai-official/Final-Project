/**
 * @file sweep.cpp
 * 
 * @brief This file contains the functions for updating values in 
 *        parallel processing arrays used for solving Heat Equation.
 *        
 * 
 * 
 * @author LI Yihai
 * @version 3.0
 * @date May 25, 2024
 */

#ifndef FINAL_PROJECT_SWEEP_HPP_LIYIHAI
#define FINAL_PROJECT_SWEEP_HPP_LIYIHAI

#include "array.cpp"

namespace final_project {

  /**
   * @brief Setup for the heat2d sweep.
   * 
   * @tparam T The type of the elements stored in the array.
   * @param coff Coefficient.
   * @param time Time step.
   */
  template <class T>
  void array2d_distribute<T>::sweep_setup_heat2d(double coff, double time)
  {
    auto min = [](auto const a, auto const b){return (a <= b) ? a : b;};

    hx = (double) (time - 0) / (double) (glob_Rows + 1);
    hy = (double) (time - 0) / (double) (glob_Cols + 1);

    dt = 0.25 * (double) (min(hx, hy) * min(hx, hy)) / coff;
    dt = min(dt, 0.1);

    weight_x = coff * dt / (hx * hx);
    weight_y = coff * dt / (hy * hy);

    diag_x = -2.0 + hx * hx / (2 * coff * dt);
    diag_y = -2.0 + hy * hy / (2 * coff * dt);
  }

  /**
   * @brief Perform the heat2d sweep.
   * 
   * 
   * @tparam T The type of the elements stored in the array.
   * @param out Output array.
   */
  template <class T>
  void array2d_distribute<T>::sweep_heat2d(array2d_distribute<T>& out)
  {
    std::size_t i,j, Nx {this->rows() - 2}, Ny{this->cols() - 2};

    for (i = 1; i <= Nx; ++i)
      for (j = 1; j <= Ny; ++j)
        out(i,j) = weight_x * ((*this)(i-1, j) + (*this)(i+1, j) + (*this)(i,j) * diag_x)
                 + weight_y * ((*this)(i, j-1) + (*this)(i, j+1) + (*this)(i,j) * diag_y);
  }
  
} // namespace final_project


#endif