#pragma once 
#include "array.cpp"
#include <iostream>

/* ------------------------------ Sweep Functions ------------------------------ */
/* Heat Equation */
template <typename T>
void sweep_Heat(final_project::Array<T> const in, final_project::Array<T>& out)
{
  int i, j;
  int nx, ny;
  
  double coff = 1;
  double dt, hx, hy;
  double diag_x, diag_y;
  double weight_x, weight_y;

  nx = (int) in.get_num_rows() - 2;
  ny = (int) in.get_num_cols() - 2;

  auto min = [](auto const a, auto const b){return (a <= b) ? a : b;};

  hx = (double) 1 / (double) (nx + 1);
  hy = (double) 1 / (double) (ny + 1);

  dt = 0.25 * (double) (min(hx, hy) * min(hx, hy)) / coff;
  dt = min(dt, 0.1);

  weight_x = coff * dt / (hx * hx);
  weight_y = coff * dt / (hy * hy);

  diag_x = -2.0 + hx * hx / (2 * coff * dt);
  diag_y = -2.0 + hy * hy / (2 * coff * dt);
  
  for (i = 1; i <= nx; ++i)
  {
    for (j = 1; j <= ny; ++j)
    {
      out(i,j) = weight_x * (in(i-1, j) + in(i+1, j) + in(i,j)*diag_x)
               + weight_y * (in(i, j-1) + in(i, j+1) + in(i,j)*diag_y);
    }
  }
}

template <typename T>
void sweep_Heat(final_project::Array<T> const in, final_project::Array<T>& out, 
                const int s[2], const int e[2])
{
  int i, j;
  int nx, ny;
  
  double coff = 1;
  double dt, hx, hy;
  double diag_x, diag_y;
  double weight_x, weight_y;

  nx = (int) in.get_num_rows() - 2;
  ny = (int) in.get_num_cols() - 2;

  auto min = [](auto const a, auto const b){return (a <= b) ? a : b;};

  hx = (double) 1 / (double) (nx + 1);
  hy = (double) 1 / (double) (ny + 1);

  dt = 0.25 * (double) (min(hx, hy) * min(hx, hy)) / coff;
  dt = min(dt, 0.1);

  weight_x = coff * dt / (hx * hx);
  weight_y = coff * dt / (hy * hy);

  diag_x = -2.0 + hx * hx / (2 * coff * dt);
  diag_y = -2.0 + hy * hy / (2 * coff * dt);
  
  for (i = s[0]; i <= e[0]; ++i)
  {
    for (j = s[1]; j <= e[1]; ++j)
    {
      out(i,j) = weight_x * (in(i-1, j) + in(i+1, j) + in(i,j)*diag_x)
               + weight_y * (in(i, j-1) + in(i, j+1) + in(i,j)*diag_y);
    }
  }
}

/* Possion Equation */
template <typename T>
void sweep_Possion(final_project::Array<T> const in, final_project::Array<T> const bias, final_project::Array<T>& out)
{
  auto nx = in.get_num_rows() - 2;
  auto ny = in.get_num_cols() - 2;

  double hx {1.0 / ((double)nx+1)};
  double hy {1.0 / ((double)ny+1)};

  std::size_t i, j;
  for (i = 1; i <= nx; ++i)
  {
    for (j = 1; j <= ny; ++j)
    {
      out(i,j) = 0.25 * ( in(i-1, j) + in(i+1, j) + 
                          in(i, j+1) + in(i, j+1) - hx * hy * bias(i,j));
    } 
  }
}

template <typename T>
void sweep_Possion(final_project::Array<T> const in, final_project::Array<T> const bias, final_project::Array<T>& out, 
                const int s[2], const int e[2])
{
  auto nx = in.get_num_rows() - 2;
  auto ny = in.get_num_cols() - 2;

  double hx {1.0 / ((double)nx+1)};
  double hy {1.0 / ((double)ny+1)};

  std::size_t i, j;

  for (i = s[0]; i <= e[0]; ++i) 
  {
    for (j = s[1]; j < e[1]+1; ++j) 
    {
      out(i,j) = 0.25 * ( in(i-1, j) + in(i+1, j) + 
                          in(i, j+1) + in(i, j+1) - hx * hy * bias(i,j));
    }
  }
}

namespace final_project {

  /**
  * @brief This version is the Pure Sweep with 2 for loops.
  *
  */
  template <typename T>
  void Array_Distribute<T>::sweep(Array_Distribute<T>& out)
  {
    int i, j;

    int nx = this->get_num_rows() - 2;
    int ny = this->get_num_cols() - 2;

    double coff = 1;
    double dt, hx, hy;
    double diag_x, diag_y;
    double weight_x, weight_y;

    auto min = [](auto const a, auto const b){return (a <= b) ? a : b;};

    hx = (double) 1 / (double) (nx_glob+1);
    hy = (double) 1 / (double) (ny_glob+1);

    dt = 0.25 * (double) (min(hx, hy) * min(hx, hy)) / coff;
    dt = min(dt, 0.1);

    weight_x = coff * dt / (hx * hx);
    weight_y = coff * dt / (hy * hy);

    diag_x = -2.0 + hx * hx / (2 * coff * dt);
    diag_y = -2.0 + hy * hy / (2 * coff * dt);


    for (i = 1; i <= nx; ++i)
      for (j = 1; j <= ny; ++j)
      {
        out(i,j) = weight_x * ((*this)(i-1, j) + (*this)(i+1, j) + (*this)(i,j) * diag_x)
                + weight_y * ((*this)(i, j-1) + (*this)(i, j+1) + (*this)(i,j) * diag_y);
      }
  }


  /**
  * @brief This version has a data racing problem
  *
  */
  template <typename T>
  void Array_Distribute<T>::sweep(Array_Distribute<T>& out, const int p_id)
  {

    int i, j, off;

    int nx = this->get_num_rows() - 2;
    int ny = this->get_num_cols() - 2;

    double coff = 1;
    double dt, hx, hy;
    double diag_x, diag_y;
    double weight_x, weight_y;

    auto min = [](auto const a, auto const b){return (a <= b) ? a : b;};

    hx = (double) 1 / (double) (nx_glob+1);
    hy = (double) 1 / (double) (ny_glob+1);

    dt = 0.25 * (double) (min(hx, hy) * min(hx, hy)) / coff;
    dt = min(dt, 0.1);

    weight_x = coff * dt / (hx * hx);
    weight_y = coff * dt / (hy * hy);

    diag_x = -2.0 + hx * hx / (2 * coff * dt);
    diag_y = -2.0 + hy * hy / (2 * coff * dt);
    
    for (i = 1; i <= nx; ++i)
    {
      off = 1 + (p_id + i + 1 ) % 2;
      for (j = off; j <= ny; j+=2)
      {
        out(i,j) = weight_x * ((*this)(i-1, j) + (*this)(i+1, j) + (*this)(i,j) * diag_x)
                + weight_y * ((*this)(i, j-1) + (*this)(i, j+1) + (*this)(i,j) * diag_y);
      }
    }
  }

  /**
  * @brief This verion is trying to solve the data racing problem.
  * 
  */
  template <typename T>
  void Array_Distribute<T>::sweep3(Array_Distribute<T>& out, const int p_id)
  {
    int i, j;

    int nx = this->get_num_rows() - 2;
    int ny = this->get_num_cols() - 2;

    double coff = 1;
    double dt, hx, hy;
    double diag_x, diag_y;
    double weight_x, weight_y;

    auto min = [](auto const a, auto const b){return (a <= b) ? a : b;};

    hx = (double) 1 / (double) (nx_glob+1);
    hy = (double) 1 / (double) (ny_glob+1);

    dt = 0.25 * (double) (min(hx, hy) * min(hx, hy)) / coff;
    dt = min(dt, 0.1);

    weight_x = coff * dt / (hx * hx);
    weight_y = coff * dt / (hy * hy);

    diag_x = -2.0 + hx * hx / (2 * coff * dt);
    diag_y = -2.0 + hy * hy / (2 * coff * dt);

    int start {(p_id == 0) ? 1 : (nx / 2)};
    int end {(p_id == 0) ? (nx / 2) : nx};

    for (i = 1; i <= nx; ++i)
      for (j = 1; j <= ny; ++j)
      {
        out(i,j) = weight_x * ((*this)(i-1, j) + (*this)(i+1, j) + (*this)(i,j) * diag_x)
                 + weight_y * ((*this)(i, j-1) + (*this)(i, j+1) + (*this)(i,j) * diag_y);
      }
  }

}