#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

#define MAX_N 30+2
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

  ~Matrix() {delete[] data;}

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
        os << std::fixed << std::setprecision(5) << std::setw(9) << in(ridx, cidx);
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
inline void store_matrix(Matrix<T> const in, std::string const fname)
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
double get_difference(Matrix<T> const ping, Matrix<T> const pong)
{
  double temp, sum = 0.0;

  std::size_t i, j;
  for (i = 1; i <= ping.get_num_rows() - 1; ++i)
  {
    for (j = 1; j <= ping.get_num_cols() - 1; ++j)
    {
      temp = (double) ping(i, j) - pong(i, j);
      sum += temp * temp;
    }
  }
  return sum;
}

template <typename T>
void sweep_Heat(Matrix<T> const in, Matrix<T>& out)
{
  double coff = 1;
  int i, j;

  auto nx = in.get_num_rows() - 2;
  auto ny = in.get_num_cols() - 2;

  double dt, hx, hy;
  double diag_x, diag_y;
  double weight_x, weight_y;

  hx = (double) 1 / (double) in.get_num_rows();
  hy = (double) 1 / (double) in.get_num_cols();

  auto min = [](auto const a, auto const b){return (a <= b) ? a : b;};
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
    // sweep_Possion(a, f, b);
    sweep_Heat(a, b);

    // sweep_Possion(b, f, a);
    sweep_Heat(b, a);

    glob_diff = get_difference(a, b);
    if (glob_diff <= tol) 
    {
      std::cout << "Convergence at " << it << " Iterations.\n";
      store_matrix(a, "output.dat");
      break;
    }
  }
  auto t2 = std::chrono::steady_clock::now();

  std::cout << "Serial : " 
  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
  << " ms\n" << std::endl;

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