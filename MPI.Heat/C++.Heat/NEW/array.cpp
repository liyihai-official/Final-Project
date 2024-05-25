/**
 * @file array.cpp
 * @brief This file contains the definition of the array classes
 *        and features for parallel processing.
 * 
 * The classes provided in this file are designed to set up basic 
 * features of a skeleton 2D and 3D array. And enhanced them into
 * distributed 2D and 3D array used in parallel processing 
 * environments.
 * 
 * 
 * @author LI Yihai
 * @version 3.0
 * @date May 25, 2024
 */

#ifndef FINAL_PROJECT_ARRAY_HPP_LIYIHAI
#define FINAL_PROJECT_ARRAY_HPP_LIYIHAI

#pragma once
#include <iostream>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <iterator>
#include <unistd.h>

#include <mpi.h>
#include <fstream>

#include "assert.cpp"

template<typename T>
MPI_Datatype get_mpi_type();

template<>
MPI_Datatype get_mpi_type<int>()    { return MPI_INT; }

template<>
MPI_Datatype get_mpi_type<float>()  { return MPI_FLOAT; }

template<>
MPI_Datatype get_mpi_type<double>() { return MPI_DOUBLE; }

/**
 * @brief Decompose a 1D problem among processors.
 * 
 * @param n The total size of the problem.
 * @param problem_size The number of processors.
 * @param rank The rank of the current processor.
 * @param s Reference to the starting index for the current processor.
 * @param e Reference to the ending index for the current processor.
 * 
 * @return int Status code.
 */
int Decomp1d(const int n, const int problem_size, const int rank, int& s, int& e)
  {
    int nlocal, deficit;
    nlocal  = n / problem_size;

    s = rank * nlocal + 1;
    deficit = n % problem_size;
    s = s + ((rank < deficit) ? rank : deficit);
    if (rank < deficit) nlocal++;
    e  = s + nlocal - 1;
    if (e > n || rank == problem_size-1) e = n;

    return 0;
  }



namespace final_project {

  /**
   * @brief A 2D array class.
   * 
   * @tparam T The type of the elements stored in the array.
   */
  template <class T>
  class array2d {
    public:
      std::size_t Rows, Cols;
      std::unique_ptr<T[]> data;
      
    public:
      array2d(std::size_t Rows, std::size_t Cols) 
        : Rows{Rows}, Cols{Cols}, data{std::make_unique<T[]>(Rows * Cols)} {}

    public:
      typedef T               value_type;
      typedef T&              reference;
      typedef const T&        const_reference;
      typedef T*              iterator;
      typedef const T*        const_iterator;
      typedef std::size_t     size_type;

      // Sizes
      const size_type rows() { return Rows; }
      const size_type cols() { return Cols; }
      const size_type size() { return Rows * Cols; }

      const size_type rows() const { return Rows; }
      const size_type cols() const { return Cols; }
      const size_type size() const { return Rows * Cols; }

      // Iterators
      iterator          begin()       { return data.get(); }
      const_iterator    begin() const { return data.get(); }
      const_iterator   cbegin() const { return data.get(); }

      iterator            end()       { return data.get() + Rows * Cols; }
      const_iterator      end() const { return data.get() + Rows * Cols; }
      const_iterator     cend() const { return data.get() + Rows * Cols; }


    public:
      // Operator ()
      reference operator() (size_type i)
      {
        return FINAL_PROJECT_ASSERT_MSG( (i < Rows * Cols), "out of range"), data[i];
      }

      reference operator() (size_type i) const
      {
        return FINAL_PROJECT_ASSERT_MSG( (i < Rows * Cols), "out of range"), data[i];
      }

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
      array2d<T>& operator= (const array2d<T2> & rhs)
      {
        std::copy(rhs.begin(), rhs.end(), begin());
        return *this;
      }
      

      // assign one value to all data
      void assign (const T& value) { fill (value); }
      void fill (const T& value)
      {
        std::fill_n(begin(), size(), value);
      }

      // Swap
      void swap (array2d<T>& other)
      {
        for (size_type i = 0; i < size(); ++i)
          data.swap(other.data);
      }

      // Resize 
      /**
       * @brief Resize the array.
       * 
       * @param new_rows The new number of rows.
       * @param new_cols The new number of columns.
       */
      void resize(size_type new_rows, size_type new_cols) {
        Rows = new_rows;
        Cols = new_cols;
        data = std::make_unique<T[]>(new_rows * new_cols);
      }

      // Friend function declarations
      template <class U>
      friend std::ostream& operator<<(std::ostream& os, const array2d<U>& in);

  }; /* class array2d */


  /**
   * @brief A distributed 2D array class.
   * 
   * @tparam T The type of the elements stored in the array.
   */
  template <class T>
  class array2d_distribute : public array2d<T> {
    public:
      std::size_t glob_Rows, glob_Cols;

      /* MPI Topology Features */
      int rank, num_proc;
      int nbr_up, nbr_down, nbr_right, nbr_left;

      constexpr static int dimension {2};
      int starts[dimension], ends[dimension], coordinates[dimension];

      MPI_Datatype vecs[dimension];
      MPI_Comm     communicator;

    public:
      array2d_distribute() : glob_Cols {0}, glob_Rows{0}, array2d<T>(0,0) {} ;

      /**
       * @brief Generate the distributed array by inputting communicator and global sizes.
       * 
       * @param gRows Global number of rows.
       * @param gCols Global number of columns.
       * @param dims Array of dimensions.
       * @param comm MPI communicator.
      */
      void distribute(std::size_t gRows, std::size_t gCols, const int dims[2], MPI_Comm comm)
      {
        communicator = comm;
        glob_Rows = gRows; glob_Cols = gCols;
        
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &num_proc);
        MPI_Cart_coords(comm, rank, dimension, coordinates);

        Decomp1d(glob_Rows-2, dims[0], coordinates[0], starts[0], ends[0]);
        Decomp1d(glob_Cols-2, dims[1], coordinates[1], starts[1], ends[1]);
        
        MPI_Cart_shift(comm, 0, 1, &nbr_up,   &nbr_down );
        MPI_Cart_shift(comm, 1, 1, &nbr_left, &nbr_right);

        /* Setup vector types of halo */ 
        const int nx {ends[0] - starts[0] + 1 + 2};
        const int ny {ends[1] - starts[1] + 1 + 2};

        MPI_Type_contiguous(ends[1] - starts[1] + 1,         get_mpi_type<T>(), &vecs[0]);
        MPI_Type_commit(&vecs[0]);

        MPI_Type_vector(    ends[0] - starts[0] + 1, 1, ny , get_mpi_type<T>(), &vecs[1]);
        MPI_Type_commit(&vecs[1]);
        
        this->resize(nx, ny);
      }
    

    // Update data
    private:
      double coff;
      double diag_x, diag_y, weight_x, weight_y;
      double dt, hx, hy;

    public:
      // Heat Equation
      void sweep_setup_heat2d(double coff, double time);
      void sweep_heat2d(array2d_distribute<T>&out);
      void sweep_heat2d_omp1(array2d_distribute<T>&out, const int p_id);

      // Possion Equation
      // void sweep_setup_possion2d();
      // void sweep_possion2d(array2d_distribute<T>&out, array2d_distribute<T> const bias);

    // exchanges, communications
    public:
      void I_exchange2d();
      void SR_exchange2d();

      void Gather2d(array2d<T>& gather, const int root, MPI_Comm comm);
  }; /* class array2d_distribute */


  /**
   * @brief I/O of array 2d.
   * 
   * @tparam T The type of elements stored in the array.
   * @param os The output stream.
   * @param in The array to output.
   * @return std::ostream& The output stream.
  */
  template <class T>
  std::ostream& operator<<(std::ostream& os, const array2d<T>& in) {
    for (std::size_t ridx = 0; ridx < in.Rows; ++ridx) {
      os << "";
      for (std::size_t cidx = 0; cidx < in.Cols; ++cidx) {
        os << std::fixed << std::setprecision(5) << std::setw(9) << in(ridx, cidx);
      }
      os << "" << std::endl;
    }
    return os;
  }

  /**
   * @brief Get the difference between two distributed arrays.
   * 
   * @tparam T The type of elements stored in the arrays.
   * @param ping The first array.
   * @param pong The second array.
   * @return double The difference.
   */
  template <class T>
  double get_difference(const array2d_distribute<T>& ping, const array2d_distribute<T>& pong)
  {
    FINAL_PROJECT_ASSERT_MSG((ping.Rows == pong.Rows && ping.Cols == pong.Cols), "Different Shape!");

    double temp, diff {0.0};
    for (std::size_t i = 0; i < ping.size(); i++)
    {
      temp = ping(i) - pong(i);
      diff += temp * temp;
    }

    return diff;
  }


  /**
   * @brief A 3D array class.
   * 
   * @tparam T The type of elements stored in the array.
   */
  template <class T>
  class array3d {
    public:
      std::size_t Rows, Cols, Height;
      std::unique_ptr<T[]> data;

    public:
      typedef T               value_type;
      typedef T&              reference;
      typedef const T&        const_reference;
      typedef T*              iterator;
      typedef const T*        const_iterator;
      typedef std::size_t     size_type;

      // Sizes
      const size_type rows()    { return Rows; }
      const size_type cols()    { return Cols; }
      const size_type height()  { return Height; }
      const size_type size()    { return Rows * Cols * Height; }

      const size_type rows()    const { return Rows; }
      const size_type cols()    const { return Cols; }
      const size_type height()  const { return Height; }
      const size_type size()    const { return Rows * Cols * Height; }

      // Iterators
      iterator          begin()       { return data.get(); }
      const_iterator    begin() const { return data.get(); }
      const_iterator   cbegin() const { return data.get(); }

      iterator            end()       { return data.get() + Rows * Cols * Height; }
      const_iterator      end() const { return data.get() + Rows * Cols * Height; }
      const_iterator     cend() const { return data.get() + Rows * Cols * Height; }

    public:
      // Operator ()
      reference operator() (size_type i)
      {
        return FINAL_PROJECT_ASSERT_MSG( (i < Rows * Cols * Height), "out of range"), data[i];
      }

      reference operator() (size_type i) const
      {
        return FINAL_PROJECT_ASSERT_MSG( (i < Rows * Cols * Height), "out of range"), data[i];
      }

      reference operator() (size_type i, size_type j, size_type k)
      {
        return FINAL_PROJECT_ASSERT_MSG( (i < Rows && j < Cols && k < Height), "out of range"), data[i * (Cols * Height) + j * Height + k];
      }

      const_reference operator() (size_type i, size_type j, size_type k) const
      {
        return FINAL_PROJECT_ASSERT_MSG( (i < Rows && j < Cols && k < Height), "out of range"), data[i * (Cols * Height) + j * Height + k];
      }

      // assignment operator = 
      template <typename T2>
      array3d<T>& operator= (const array3d<T2> & rhs)
      {
        std::copy(rhs.begin(), rhs.end(), begin());
        return *this;
      }
      

      // assign one value to all data
      void assign (const T& value) { fill (value); }
      void fill (const T& value)
      {
        std::fill_n(begin(), size(), value);
      }

      // Swap
      void swap (array3d<T>& other)
      {
        for (size_type i = 0; i < size(); ++i)
          data.swap(other.data);
      }

      // Resize 
      /**
       * @brief Resize the array.
       * 
       * @param new_rows The new number of rows.
       * @param new_cols The new number of columns.
       * @param new_Height The new height.
       */
      void resize(size_type new_rows, size_type new_cols, size_type new_Height) {
        Rows = new_rows;
        Cols = new_cols;
        Height = new_Height;
        data = std::make_unique<T[]>(Rows * Cols * Height);
      }

      // Friend function declarations
      template <class U>
      friend std::ostream& operator<<(std::ostream& os, const array3d<U>& in);
    
  };


  /**
   * @brief I/O of array 3d.
   * 
   * @tparam T The type of elements stored in the array.
   * @param os The output stream.
   * @param in The array to output.
   * @return std::ostream& The output stream.
   */
  template <class T>
  std::ostream& operator<<(std::ostream& os, const array3d<T>& in) {
    for (std::size_t ridx = 0; ridx < in.Rows; ++ridx) {
      os << "";
      for (std::size_t cidx = 0; cidx < in.Cols; ++cidx) {
        for (std::size_t kidx = 0; kidx < in.Height; ++kidx) {
          os << std::fixed << std::setprecision(5) << std::setw(9) << in(ridx, cidx, kidx);
        }
        os << "" << std::endl;
      }
      os << "" << std::endl;
    }
    return os;
  }


  /**
   * @brief Print the 2D array in order according to the rank of each processor.
   * 
   * This function ensures that each processor prints its portion of the 2D array in order,
   * based on their rank. It uses MPI barriers to synchronize the printing process and 
   * small sleeps to allow ordered output.
   * 
   * @tparam T The type of the elements in the array.
   * @param in The distributed 2D array to be printed.
   */
  template <class T>
  void print_in_order(final_project::array2d_distribute<T>& in)
  {
    MPI_Barrier(in.communicator);
    std::cout << "Attempting to print 2d array in order" << std::endl;
    sleep(0.01);
    MPI_Barrier(in.communicator);

    for ( int i = 0; i < in.num_proc; ++i)
    {
      if ( i == in.rank )
      {
        std::cout << "proc : " << in.rank << " at " 
        << "( "<< in.coordinates[0] << ", "<< in.coordinates[1] << " )"
        << "\n" << in << std::endl;
      }
      fflush(stdout);
      sleep(0.01);
      MPI_Barrier(in.communicator);
    }
  }

} // namespace final_project



#endif /* FINAL_PROJECT_ARRAY_HPP */