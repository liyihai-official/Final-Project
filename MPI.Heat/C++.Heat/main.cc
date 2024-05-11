#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

#define MAX_N 12+2
#define MAX_it 2000

template <typename T>
class Matrix {
  public:
  Matrix() = delete;
  Matrix(std::size_t const rows, std::size_t const cols) 
  : rows{rows}, cols{cols} 
  {data = new T[rows * cols];}

  Matrix(Matrix const& other)
  : rows{other.rows}, cols{other.cols}, data{new T[rows * cols]}
  {std::copy(other.data, other.data + rows * cols, data);}
  
  Matrix(Matrix &&);

  Matrix& operator=(const Matrix& other)
  {
    if (this != &other)
    {
      delete[] data;
      rows = other.rows;
      cols = other.cols;
      data = new T[rows * cols];
      std::copy(other.data, other.data + rows*cols, data);
    }
    return *this;
  }
  
  Matrix& operator=(Matrix&& other)
  {
    if (this != &other)
    {
      delete[] data;
      rows = other.rows;
      cols = other.cols;
      data = other.data;

      other.date = nullptr;
    }

    return *this;
  }

  ~Matrix() 
  {
    delete[] data;
  }

  std::size_t get_num_rows() const {return rows;}
  std::size_t get_num_cols() const {return cols;}

  T operator() (std::size_t const ridx, std::size_t const cidx) const
  {
    if (ridx > rows || cidx > cols) {
      std::string msg {"Matrix subscript out of index.\n"};
      throw std::out_of_range(msg);
    }

    return data[ridx * cols + cidx];
  }

  T& operator() (std::size_t const ridx, std::size_t const cidx)
  {
    if (ridx > rows || cidx > cols) {
      std::string msg {"Matrix subscript out of index.\n"};
      throw std::out_of_range(msg);
    }

    return data[ridx * cols + cidx];
  }

  
  friend std::ostream& operator<< (std::ostream& os, Matrix<T> const & in)
  {
    std::size_t ridx, cidx, rows = in.get_num_rows(), cols = in.get_num_cols();

    for (ridx = 0; ridx < in.rows; ++ridx) {
      os << "| ";
      for (cidx = 0; cidx < in.cols; ++cidx) 
      {
        os << std::fixed << std::setprecision(3) << std::setw(9) << in(ridx, cidx);
      }
      os << " |" << std::endl;
    }
    return os;
  } 


  private:
    std::size_t rows, cols;
    T* data;
};


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
      out(i,j) = 0.25 * ( in(i - 1, j) + in( i + 1, j) + 
                          in(i, j + 1) + in( i, j + 1) - hx * hy * bias(i,j));
    } 
  }
}

template <typename T>
void init_conditions(Matrix<T>& init, Matrix<T>& init_other, Matrix<T>& bias)
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
        init(i, j) = 1;
        init_other(i, j) = 1;
      }

      if (i == NX)
      {
        init(i, j) = 2;
        init_other(i, j) = 2;
      }

      if (j == 0) 
      {
        init(i, j) = 3;
        init_other(i, j) = 3;
      }

      if (j == NY)
      {
        init(i, j) = 4;
        init_other(i, j) = 4;
      }
    } 
  }
}


int main(int argc, char ** argv)
{
  Matrix<double> a(MAX_N, MAX_N), b(MAX_N, MAX_N), f(MAX_N, MAX_N), solution(MAX_N, MAX_N), gather(MAX_N, MAX_N);

  int nx, ny, it;
  double loc_diff, glob_diff, loc_err, glob_err, tol = 1E-15;


  init_conditions(a, b, f);
  

  auto t1 = std::chrono::steady_clock::now();
  for (it = 0; it < MAX_it; ++it)
  {
    sweep_Possion(a, f, b);
    sweep_Possion(b, f, a);
  }
  auto t2 = std::chrono::steady_clock::now();

  std::cout << "Serial : " 
  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
  << " ms\n" << std::endl;

  std::cout << a;

  return EXIT_SUCCESS;
}