#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <cassert>
#include <mpi.h>

#define BOOST_ASSERT_MSG(expr, msg) assert((expr) && (msg))
#define FINAL_PROJECT_ASSERT_MSG(expr, msg) assert((expr) && (msg))

namespace final_project {

  template <class T, std::size_t Rows, std::size_t Cols>
  class array2d {
    public:
      std::unique_ptr<T[]> data;
      
    public:
      array2d() : data(std::make_unique<T[]>(Rows * Cols)) {}

    public:
      typedef T               value_type;
      typedef T&              reference;
      typedef const T&        const_reference;
      typedef T*              iterator;
      typedef const T*        const_iterator;
      typedef std::size_t     size_type;

      // Sizes
      static constexpr size_type rows() { return Rows; }
      static constexpr size_type cols() { return Cols; }
      static constexpr size_type size() { return Rows * Cols; }

      // Iterators
      iterator          begin()       { return data.get(); }
      const_iterator    begin() const { return data.get(); }
      const_iterator   cbegin() const { return data.get(); }

      iterator            end()       { return data.get() + Rows * Cols; }
      const_iterator      end() const { return data.get() + Rows * Cols; }
      const_iterator     cend() const { return data.get() + Rows * Cols; }


    public:
      // Operator ()
      reference operator() (size_type i, size_type j)
      {
        return FINAL_PROJECT_ASSERT_MSG( (i < Rows && j < Cols), "out of range"), data[i * Cols + j];
      }

      const_reference operator() (size_type i, size_type j) const
      {
        return FINAL_PROJECT_ASSERT_MSG( (i < Rows && j < Cols), "out of range"), data[i * Cols + j];
      }

      // assignment operator = 
      template <typename T2>
      array2d<T, Rows, Cols>& operator= (const array2d<T2, Rows, Cols> & rhs)
      {
        std::copy(rhs.begin(), rhs.end(), begin());
      }
      

      // assign one value to all data
      void assign (const T& value) { fill (value); }
      void fill (const T& value)
      {
        std::fill_n(begin(), size(), value);
      }

      // Swap
      void swap (array2d<T, Rows, Cols>& other)
      {
        for (size_type i = 0; i < size(); ++i)
          data.swap(other.data);
      }

      // Friend function declarations
      template <class U, std::size_t R, std::size_t C>
      friend std::ostream& operator<<(std::ostream& os, const array2d<U, R, C>& in);

  };

  template <class T, std::size_t Rows, std::size_t Cols>
  std::ostream& operator<<(std::ostream& os, const array2d<T, Rows, Cols>& in) {
    for (std::size_t ridx = 0; ridx < Rows; ++ridx) {
      os << "";
      for (std::size_t cidx = 0; cidx < Cols; ++cidx) {
        os << std::fixed << std::setprecision(5) << std::setw(9) << in(ridx, cidx);
      }
      os << "" << std::endl;
    }
    return os;
  }

} // namespace final_project


namespace final_project {
  template <class T, std::size_t Rows, std::size_t Cols>
  class array2d_distribute : public array2d<T, Rows, Cols> {
    public:
      std::size_t glob_Rows, glob_Cols;

      /* MPI Topology Features */
      int rank, num_proc;
      int nbr_up, nbr_down, nbr_right, nbr_left;

      constexpr static int dimension {2};
      int starts[dimension], ends[dimension], coordinates[dimension];

    public:
      array2d_distribute() : array2d<T, Rows, Cols>() {}

    public:
      void distribute(std::size_t gRows, std::size_t gCols, MPI_Comm comm_cart)
      {
        
      }
  };
} // namespace final_project

int main (int argc, char ** argv)
{
  final_project::array2d<double, 5, 3> A;
  A.fill(10);
  // std::cout << A;

  final_project::array2d_distribute<double, 5, 3> a;
  a.fill(5);
  // std::cout << a;

  MPI_Init(&argc, &argv);
  

  
  return 0;
}