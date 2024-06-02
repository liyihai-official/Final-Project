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

#include "multi_array/array.cpp"
#include <omp.h>

namespace final_project {

////////////////////////////////////////////////////////////////////////////////////////////////////////////
//          Heat 1D
////////////////////////////////////////////////////////////////////////////////////////////////////////////

  template <class T>
  void array1d_distribute<T>::sweep_setup_heat1d(double coff, double time)
  {
    FINAL_PROJECT_ASSERT_HPP_LIYIHAI((glob_N != 0), "Invalid Distribute array1d. Run distribute first.");

    hx = (double) (time - 0) / (double) (glob_N + 1);

    dt = 0.5 * hx * hx / coff;
    dt = std::min(dt, 0.1);

    weight = coff * dt / (hx * hx);

    diag = -2.0 + hx * hx / (dimension * coff * dt);
  }

  template <class T>
  void array1d_distribute<T>::sweep_heat1d(array1d_distribute<T> &out)
  {
    std::size_t i, N {this->size() - 2};

    for (i = 1; i <= N; ++i)
    {
      double current = (*this)(i+1);
      out(i) =  weight*((*this)(i-1) + (*this)(i+1)) 
              + current * diag * weight;
    }
  }

////////////////////////////////////////////////////////////////////////////////////////////////////////////
//          Heat 2D
////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
    FINAL_PROJECT_ASSERT_MSG((glob_Rows != 0 && glob_Cols != 0), "Invalid Distribute array2d. Run distribute first.");

    hx = (double) (time - 0) / (double) (glob_Rows + 1);
    hy = (double) (time - 0) / (double) (glob_Cols + 1);

    dt = 0.25 * std::min({hx, hy}) * std::min({hx, hy}) / coff;
    dt = std::min(dt, 0.1);

    weight_x = coff * dt / (hx * hx);
    weight_y = coff * dt / (hy * hy);

    diag_x = -2.0 + hx * hx / (dimension * coff * dt);
    diag_y = -2.0 + hy * hy / (dimension * coff * dt);
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
      {
        double current = (*this)(i, j);
        out(i,j) = weight_x * ((*this)(i-1, j) + (*this)(i+1, j))
                 + weight_y * ((*this)(i, j-1) + (*this)(i, j+1))
                 + current  * (diag_x*weight_x + diag_y*weight_y);
        // out(i,j) = weight_x * ((*this)(i-1, j) + (*this)(i+1, j) + (*this)(i,j) * diag_x)
        //          + weight_y * ((*this)(i, j-1) + (*this)(i, j+1) + (*this)(i,j) * diag_y);
      }
  }

  /**
   * @brief Perform the heat2d sweep, with OpenMP features.
   * 
   * 
   * @tparam T The type of the elements stored in the array.
   * @param out Output array.
   */
  template <class T>
  void array2d_distribute<T>::sweep_heat2d_omp1(array2d_distribute<T>& out, const int p_id)
  {
    std::size_t i,j,off, Nx {this->rows() - 2}, Ny{this->cols() - 2};

    #pragma omp private(i, j, off)
    for (i = 1; i <= Nx; ++i)
    {
      off = 1 + (p_id + i + 1) % 2;
      for (j = off; j <= Ny; j+=2)
      {
        out(i,j) = weight_x * ((*this)(i-1, j) + (*this)(i+1, j) + (*this)(i,j) * diag_x)
                 + weight_y * ((*this)(i, j-1) + (*this)(i, j+1) + (*this)(i,j) * diag_y);
      }
    }
  }
  
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//          Heat 3D
////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Setup for the heat3d sweep.
   * 
   * @tparam T The type of the elements stored in the array.
   * @param coff Coefficient.
   * @param time Time step.
   */
  template <class T>
  void array3d_distribute<T>::sweep_setup_heat3d(double coff, double time)
  {
    FINAL_PROJECT_ASSERT_MSG((glob_Rows != 0 && glob_Cols != 0 && glob_Heights != 0), "Invalid Distribute array3d. Run distribute first.");

    hx = (double) (time - 0) / (double) (glob_Rows + 1);
    hy = (double) (time - 0) / (double) (glob_Cols + 1);
    hz = (double) (time - 0) / (double) (glob_Heights + 1);

    dt = 0.125 * std::min({hx, hy, hz}) * std::min({hx, hy, hz}) / coff;
    dt = std::min(dt, 0.1);

    weight_x = coff * dt / (hx * hx);
    weight_y = coff * dt / (hy * hy);
    weight_z = coff * dt / (hz * hz);

    diag_x = -2.0 + hx * hx / (dimension * coff * dt);
    diag_y = -2.0 + hy * hy / (dimension * coff * dt);
    diag_z = -2.0 + hz * hz / (dimension * coff * dt);

  }

  /**
   * @brief Perform the heat3d sweep.
   * 
   * 
   * @tparam T The type of the elements stored in the array.
   * @param out Output array.
   */
  template <class T>
  void array3d_distribute<T>::sweep_heat3d(array3d_distribute<T>& out)
  {
    
    std::size_t i,j,k;
    std::size_t Nx {this->rows() - 2}, Ny{this->cols() - 2},  Nz{this->height() - 2};

    for (i = 1; i <= Nx; ++i) 
        for (j = 1; j <= Ny; ++j) 
            for (k = 1; k <= Nz; ++k) 
            {
                double current = (*this)(i, j, k);
                out(i, j, k) =  weight_x * ((*this)(i-1, j, k) + (*this)(i+1, j, k))
                              + weight_y * ((*this)(i, j-1, k) + (*this)(i, j+1, k))
                              + weight_z * ((*this)(i, j, k-1) + (*this)(i, j, k+1))
                              + current  * (diag_x*weight_x + diag_y*weight_y + diag_z*weight_z);
            }

  }

} // namespace final_project


#endif