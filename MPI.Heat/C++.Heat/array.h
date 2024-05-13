#pragma once
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <algorithm>
#include <fstream>

template <typename T>
class Array {

  public:
  Array() = delete;
  Array(std::size_t const rows, std::size_t const cols)
  : rows{rows}, cols{cols}, data {std::vector<T> (rows*cols)} {}

  Array(std::size_t const rows, std::size_t const cols, T junk_val)
  : rows{rows}, cols{cols} 
  {
    data = std::vector<T> (rows*cols);
    for (size_t i = 0; i < rows; ++i) 
      for (size_t j = 0; j < cols; ++j) 
        data[i * cols + j] = junk_val;
  }

  ~Array() = default;

  Array(Array const& other) 
  : rows{other.rows}, cols{other.cols}, data{other.data} {}

  Array(Array && other) noexcept
  : rows{other.rows}, cols{other.cols}, data{other.data} 
  {
    other.rows = 0;
    other.cols = 0;
  }

  Array& operator=(const Array& other) {
    if (this != &other) {  
      Array tmp(other);  
      std::swap(rows, tmp.rows);
      std::swap(cols, tmp.cols);
      std::swap(data, tmp.data);
    }
    return *this;
  }

  Array& operator=(Array&& other) noexcept {
    if (this != &other) {  
      data = std::move(other.data); 
      rows = other.rows;
      cols = other.cols;

      other.rows = 0; 
      other.cols = 0;
    }
    return *this;
  }

  T  operator() (const std::size_t ridx, const std::size_t cidx) const  
  {
    if (ridx > rows || cidx > cols) {
      std::string msg {"Matrix subscript out of index.\n"};
      throw std::out_of_range(msg);
    }
    return data[ridx * cols + cidx];
  }

  T& operator() (const std::size_t ridx, const std::size_t cidx)        
  {
    if (ridx > rows || cidx > cols) {
      std::string msg {"Matrix subscript out of index.\n"};
      throw std::out_of_range(msg);
    }
    return data[ridx * cols + cidx];
  }

  friend std::ostream& operator<< (std::ostream& os, Array<T> const & in)
  {
    std::size_t ridx, cidx, rows = in.get_num_rows(), cols = in.get_num_cols();

    for (ridx = 0; ridx < in.rows; ++ridx) {
      os << "";
      for (cidx = 0; cidx < in.cols; ++cidx) 
      {
        os << std::fixed << std::setprecision(5) << std::setw(9) << in(ridx, cidx);
      }
      os << "" << std::endl;
    }
    return os;
  } 

  std::size_t get_num_rows() const {return rows;}
  std::size_t get_num_cols() const {return cols;}

  private:
  std::vector<T> data;
  std::size_t rows, cols;

};

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
inline void store_Array(Array<T> const in, std::string const fname)
{
  std::ofstream file(fname);
  if (!file.is_open()) {
      throw std::runtime_error("Unable to open file");
  }

  std::size_t i, j;
  for (i = 0; i < in.get_num_rows(); ++i) {
    for (j = 0; j < in.get_num_cols(); ++j) {
      file << std::fixed << std::setprecision(11) << std::setw(14) << in(i, j);
      if (j != in.get_num_cols() - 1) {
        file << " ";
      }
    }
    file << "\n";
  }
  file.close();
}

template <typename T>
double get_difference(Array<T> const ping, Array<T> const pong)
{
  double temp, sum = 0.0;

  std::size_t i, j;
  for (i = 1; i <= ping.get_num_rows() - 1; ++i)
  {
    for (j = 1; j <= ping.get_num_cols() - 1; ++j)
    {
      temp = ping(i, j) - pong(i, j);
      sum += temp * temp;
    }
  }
  return sum;
}

template <typename T>
double get_difference(Array<T> const ping, Array<T> const pong, 
                      const int s[2], const int e[2])
{
  double temp, sum = 0.0;

  std::size_t i, j;
  // std::cout << s[0] << "\t" << e[0] << std::endl;
  for (i = s[0]; i <= e[0]; ++i)
  {
    for (j = s[1]; j <= e[1]; ++j)
    {
      temp = (double) (ping(i, j) - pong(i, j));
      // std::cout << i <<", "<< j << "\t" << ping(i, j) << "\t" << pong(i, j) << temp * temp << "\n";
      sum = sum + temp * temp;
    }
  }
  
  return sum;
}

int Decomp1d(const int n, const int problem_size, const int rank, int& s, int& e)
{
  int nlocal, deficit;
  nlocal  = n / problem_size;

  s  = rank * nlocal + 1;
  deficit = n % problem_size;
  s  = s + ((rank < deficit) ? rank : deficit);
  if (rank < deficit) nlocal++;
  e      = s + nlocal - 1;
  if (e > n || rank == problem_size-1) e = n;

  return 0;
}