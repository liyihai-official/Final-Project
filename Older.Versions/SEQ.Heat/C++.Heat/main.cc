#include <iostream>
#include <chrono>

#include "matrix.h"

#define MAX_N 100+2
#define MAX_it 100000000


template <typename T>
void sweep_Heat(Matrix<T> const in, Matrix<T>& out)
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
void sweep_Possion(Matrix<T> const in, Matrix<T> const bias, Matrix<T>& out)
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


int main(int argc, char ** argv)
{
  Matrix<double> a(MAX_N, MAX_N), b(MAX_N, MAX_N), f(MAX_N, MAX_N), solution(MAX_N, MAX_N), gather(MAX_N, MAX_N);

  int nx, ny, it;
  double loc_diff, glob_diff, loc_err, glob_err, tol = 1E-15;

  /* -------------------------------------- Serial Ver. ------------------------------------------ */
  init_conditions(a, b, f);
  
  auto t1 = std::chrono::steady_clock::now();
  for (it = 0; it < MAX_it; ++it)
  {
    #ifdef POSSION
    sweep_Possion(a, f, b);

    sweep_Possion(b, f, a);
    #endif

    #ifdef HEAT
    sweep_Heat(a, b);

    sweep_Heat(b, a);
    #endif

    glob_diff = get_difference(a, b);
    if (glob_diff <= tol) 
    {
      std::cout << "Convergence at " << it << " Iterations.\n";
      break;
    }
  }
  auto t2 = std::chrono::steady_clock::now();

  std::cout << "Serial : " 
  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
  << " ms\n" << std::endl;

  store_matrix(a, "output_uniqueO3.dat");

  /* ------------------------------------ Parallel Ver. ------------------------------------------ */
  init_conditions(a, b, f);
  t1 = std::chrono::steady_clock::now();
  {

  }
  t2 = std::chrono::steady_clock::now();
  std::cout << "Parallel : " 
  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
  << " ms\n" << std::endl;

  return EXIT_SUCCESS;
}