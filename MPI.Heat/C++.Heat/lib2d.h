#pragma once
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <algorithm>
#include <fstream>
#include "array.h"

/* ---------------------- Initialize Conditions Functions ---------------------- */
template <typename T>
void init_conditions(Array<T>& init, Array<T>& init_other, Array<T>& bias)
{
  auto NX = init.get_num_rows() - 1;
  auto NY = init.get_num_cols() - 1;

  std::size_t i, j;
  for (i = 0; i <= NX; ++i)
  {
    for (j = 0; j <= NY; ++j)
    {
      bias(i, j) = 0;
      if (i == 0)
      {
        init(i, j) = 10;
        init_other(i, j) = 10;
      }

      if (i == NX)
      {
        init(i, j) = 10;
        init_other(i, j) = 10;
      }

      if (j == 0) 
      {
        init(i, j) = 10;
        init_other(i, j) = 10;
      }

      if (j == NY)
      {
        init(i, j) = 0;
        init_other(i, j) = 0;
      }
    } 
  }
}

template <typename T>
void twodinit_basic_Heat(Array<T>& init, Array<T>& init_other, Array<T>& bias, 
                        const int s[2], const int e[2])
{
  int i, j;
  int nx = init.get_num_rows() - 2;
  int ny = init.get_num_cols() - 2;

  for (i = s[0]-1; i <= e[0]+1; ++i) 
    for (j = s[1]-1; j <= e[1]+1; ++j)
    {
      init(i, j) = 0;
      init_other(i, j) = 0;
      bias(i, j) = 0;
    }

  /* Left Side */
  if (s[0] == 1) {
    for (j = s[1]; j < e[1]+1; ++j) {
      double yy = (double) j / (ny+1);
      init(0,       j) = 10;
      init_other(0, j) = 10;
    }
  }

  /* Right Side */
  if (e[0] == nx) {
    for (j = s[1]; j < e[1]+1; ++j) {
      double yy = (double) j / (ny+1);
      init(nx+1,        j) = 10;
      init_other(nx+1,  j) = 10;
    }
  }

  /* Bottom side */
  if (s[1] == 1) {
    for (i = s[0]; i <= e[0]; ++i) {
      init(i,       0) = 10;
      init_other(i, 0) = 10;
    }
  }
  
  /* UP side */
  if (e[1] == ny) {
    for (i = s[0]; i <= e[0]; ++i) {
      double xx = (double) i / (nx+1);
      init(i,       ny+1) = 0;
      init_other(i, ny+1) = 0;
    }
  }
}

/* Possion */
template <typename T>
void twodinit_basic_Possion(Array<T>& init, Array<T>& init_other, Array<T>& bias, 
                            const int s[2], const int e[2])
{
  int i, j;
  int nx = init.get_num_rows() - 2;
  int ny = init.get_num_cols() - 2;

  for (i = s[0]-1; i <= e[0]+1; ++i) 
    for (j = s[1]-1; j <= e[1]+1; ++j)
    {
      init(i, j) = 0;
      init_other(i, j) = 0;
      bias(i, j) = 0;
    }

  /* Left Side */
  if (s[0] == 1) {
    for (j = s[1]; j < e[1]+1; ++j) {
      double yy = (double) j / (ny+1);
      init(0,       j) = yy / (1+yy*yy);
      init_other(0, j) = yy / (1+yy*yy);
    }
  }

  /* Right Side */
  if (e[0] == nx) {
    for (j = s[1]; j < e[1]+1; ++j) {
      double yy = (double) j / (ny+1);
      init(nx+1,        j) = yy / (4+yy*yy);
      init_other(nx+1,  j) = yy / (4+yy*yy);
    }
  }

  /* Bottom side */
  if (s[1] == 1) {
    for (i = s[0]; i <= e[0]; ++i) {
      init(i,       0) = 0;
      init_other(i, 0) = 0;
    }
  }
  
  /* UP side */
  if (e[1] == ny) {
    for (i = s[0]; i <= e[0]; ++i) {
      double xx = (double) i / (nx+1);
      init(i,       ny+1) = 1 / ((xx+1)*(xx+1) + 1);
      init_other(i, ny+1) = 1 / ((xx+1)*(xx+1) + 1);
    }
  }
}

/* ------------------------------ Sweep Functions ------------------------------ */
/* Heat Equation */
template <typename T>
void sweep_Heat(Array<T> const in, Array<T>& out)
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
void sweep_Heat(Array<T> const in, Array<T>& out, 
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
void sweep_Possion(Array<T> const in, Array<T> const bias, Array<T>& out)
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
void sweep_Possion(Array<T> const in, Array<T> const bias, Array<T>& out, 
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