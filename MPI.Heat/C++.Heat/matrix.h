#include <fstream>
#include <memory>
#include <iomanip>
#include <vector>

template <typename T>
class Array : std::vector<T> {

  public:
  Array() = delete;
  Array(std::size_t const rows, std::size_t const cols)
  : rows{rows}, cols{cols} {};

  ~Array() = default;
  T  operator() (const std::size_t ridx, const std::size_t cidx) const  {return data[ridx][cidx];}
  T& operator() (const std::size_t ridx, const std::size_t cidx)        {return data[ridx][cidx];}

  private:
  std::vector<std::vector<T>> data;
  std::size_t rows, cols;

};

template <typename T>
class Matrix {
  public:
  Matrix() = delete;
  Matrix(std::size_t const rows, std::size_t const cols) 
  : rows{rows}, cols{cols} 
  {data = std::make_unique<T[]>(rows * cols);}
  
  

  Matrix(Matrix const& other)
  : rows{other.rows}, cols{other.cols}, data{std::make_unique<T[]>(rows * cols)} 
  {std::copy(other.data.get(), other.data.get() + rows * cols, data.get());}


  
  Matrix(Matrix &&);

  Matrix& operator=(const Matrix& other)
  {
    if (this != &other)
    {
      data.reset();

      rows = other.rows;
      cols = other.cols;

      data = std::make_unique<T[]>(rows * cols);
      std::copy(other.data.get(), other.data.get() + rows * cols, data.get());
      
    }
    return *this;
  }
  
  Matrix& operator=(Matrix&& other)
  {
    if (this != &other)
    {
      data.reset();

      rows = other.rows;
      cols = other.cols;

      data = std::move(other.data);

      other.rows = 0;
      other.cols = 0;
    }

    return *this;
  }

  ~Matrix() {data.reset();}

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
    std::unique_ptr<T[]> data;
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