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
   * @brief A 1D array class.
   * 
   * @tparam T The type of the elements stored in the array.
  */
  template <class T>
  class array1d {
    public:
      std::size_t N;
      std::unique_ptr<T[]> data;

    public:
      array1d(std::size_t Num) 
        : N {Num}, data{std::make_unique<T[]>(Num)} {}
    
    public:
      typedef T               value_type;
      typedef T&              reference;
      typedef const T&        const_reference;
      typedef T*              iterator;
      typedef const T*        const_iterator;
      typedef std::size_t     size_type;

      // Sizes
      const size_type size() { return N; }
      const size_type size() const { return N; }

      // Iterators
      iterator          begin()       { return data.get(); }
      const_iterator    begin() const { return data.get(); }
      const_iterator   cbegin() const { return data.get(); }

      iterator            end()       { return data.get() + N; }
      const_iterator      end() const { return data.get() + N; }
      const_iterator     cend() const { return data.get() + N; }

      // Operator ()
      reference operator() (size_type i)
      {
        return FINAL_PROJECT_ASSERT_MSG( (i < N), "out of range"), data[i];
      }

      reference operator() (size_type i) const
      {
        return FINAL_PROJECT_ASSERT_MSG( (i < N), "out of range"), data[i];
      }

      // assignment operator = 
      template <typename T2>
      array1d<T>& operator= (const array1d<T2> & rhs)
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
      void swap (array1d<T>& other)
      {
        for (size_type i = 0; i < size(); ++i)
          data.swap(other.data);
      }  
  };


  /**
   * @brief A 2D array class.
   * 
   * @tparam T The type of the elements stored in the array.
   */
  template <class T>
  class array2d : public array1d<T> {
    public:
      std::size_t Rows, Cols;
      // array1d<T> data;
      // std::unique_ptr<T[]> data;
      
    public:
      array2d(std::size_t Rows, std::size_t Cols) 
        : Rows{Rows}, Cols{Cols}, array1d<T>(Rows * Cols) {}

    public:
      using typename array1d<T>::value_type;
      using typename array1d<T>::reference;
      using typename array1d<T>::const_reference;
      using typename array1d<T>::iterator;
      using typename array1d<T>::const_iterator;
      using typename array1d<T>::size_type;

      // Sizes
      const size_type rows() { return Rows; }
      const size_type cols() { return Cols; }

      const size_type rows() const { return Rows; }
      const size_type cols() const { return Cols; }

      using array1d<T>::size; // Use size from array1d

      // Iterators
      using array1d<T>::begin; // Inherit begin from array1d
      using array1d<T>::end; // Inherit end from array1d
      using array1d<T>::cbegin; // Inherit cbegin from array1d
      using array1d<T>::cend; // Inherit cend from array1d


    public:

      // Operator ()
      reference operator() (size_type i)
      {
        return FINAL_PROJECT_ASSERT_MSG( (i < Rows * Cols), "out of range"), this->data[i];
      }

      const_reference operator() (size_type i) const
      {
        return FINAL_PROJECT_ASSERT_MSG( (i < Rows * Cols), "out of range"), this->data[i];
      }

      reference operator() (size_type i, size_type j)
      {
        return FINAL_PROJECT_ASSERT_MSG( (i < Rows && j < Cols), "out of range"), this->data[i * Cols + j];
      }

      const_reference operator() (size_type i, size_type j) const
      {
        return FINAL_PROJECT_ASSERT_MSG( (i < Rows && j < Cols), "out of range"), this->data[i * Cols + j];
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
          this->data.swap(other.data);
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
        this->data = std::make_unique<T[]>(new_rows * new_cols);
        this->N = new_rows * new_cols;
      }

      // Friend function declarations
      template <class U>
      friend std::ostream& operator<<(std::ostream& os, const array2d<U>& in);

      // Save the array to a binary file
      /**
       * @brief save 2D array to Binary File
       * 
       * @details This version is aiming to maintain the continuity of program
       * aligned with 3D version.
       * 
       * @param std::string& filename The filename
      */
      void saveToBinaryFile(const std::string& filename) const {
          std::ofstream ofs(filename, std::ios::binary);
          if (!ofs) {
              throw std::runtime_error("Cannot open file");
          }
          ofs.write(reinterpret_cast<const char*>(&Rows), sizeof(Rows));
          ofs.write(reinterpret_cast<const char*>(&Cols), sizeof(Cols));
          ofs.write(reinterpret_cast<const char*>(this->data.get()), this->size() * sizeof(T));
      }
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

    // Finite Difference Method (FDM)
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
  class array3d : array1d<T> {
    public:
      std::size_t Rows, Cols, Height;

    public:
      array3d(std::size_t Rows, std::size_t Cols, std::size_t Height) 
        : Rows{Rows}, Cols{Cols}, Height{Height}, array1d<T>(Rows * Cols * Height) {}

    public:
      using typename array1d<T>::value_type;
      using typename array1d<T>::reference;
      using typename array1d<T>::const_reference;
      using typename array1d<T>::iterator;
      using typename array1d<T>::const_iterator;
      using typename array1d<T>::size_type;

      // Sizes
      const size_type rows()    { return Rows; }
      const size_type cols()    { return Cols; }
      const size_type height()  { return Height; }

      const size_type rows()    const { return Rows; }
      const size_type cols()    const { return Cols; }
      const size_type height()  const { return Height; }

      using array1d<T>::size; // Use size from array1d

      // Iterators
      using array1d<T>::begin; // Inherit begin from array1d
      using array1d<T>::end; // Inherit end from array1d
      using array1d<T>::cbegin; // Inherit cbegin from array1d
      using array1d<T>::cend; // Inherit cend from array1d

    public:
      // Operator ()
      reference operator() (size_type i)
      {
        return FINAL_PROJECT_ASSERT_MSG( (i < Rows * Cols * Height), "out of range"), this->data[i];
      }

      reference operator() (size_type i) const
      {
        return FINAL_PROJECT_ASSERT_MSG( (i < Rows * Cols * Height), "out of range"), this->data[i];
      }

      reference operator() (size_type i, size_type j, size_type k)
      {
        return FINAL_PROJECT_ASSERT_MSG( (i < Rows && j < Cols && k < Height), "out of range"), this->data[i * (Cols * Height) + j * Height + k];
      }

      const_reference operator() (size_type i, size_type j, size_type k) const
      {
        return FINAL_PROJECT_ASSERT_MSG( (i < Rows && j < Cols && k < Height), "out of range"), this->data[i * (Cols * Height) + j * Height + k];
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
          this->data.swap(other.data);

      }

      // Resize 
      /**
       * @brief Resize the array.
       * 
       * @param new_rows The new number of rows.
       * @param new_cols The new number of columns.
       * @param new_Height The new height.
       */
      void resize(size_type new_rows, size_type new_cols, size_type new_height) {
        Rows = new_rows;
        Cols = new_cols;
        Height = new_height;
        this->N = new_rows * new_cols * new_height;
        this->data = std::make_unique<T[]>(this->N);
      }

      // Friend function declarations
      template <class U>
      friend std::ostream& operator<<(std::ostream& os, const array3d<U>& in);
    
      // Save the array to a binary file
      /**
       * @brief save 3D array to Binary File
       * 
       * @details This version is aiming to solve the problem of 
       * MATLAB does not support ASCII format in saving 3D matrix
       * 
       * @param std::string& filename The filename
      */
      void saveToBinaryFile(const std::string& filename) const {
          std::ofstream ofs(filename, std::ios::binary);
          if (!ofs) {
              throw std::runtime_error("Cannot open file");
          }
          ofs.write(reinterpret_cast<const char*>(&Rows), sizeof(Rows));
          ofs.write(reinterpret_cast<const char*>(&Cols), sizeof(Cols));
          ofs.write(reinterpret_cast<const char*>(&Height), sizeof(Height));
          ofs.write(reinterpret_cast<const char*>(this->data.get()), this->size() * sizeof(T));
      }
  };  /* class array3d */


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
          os << std::fixed << std::setprecision(5) << std::setw(20) << &in(ridx, cidx, kidx);
        }
        os << "" << std::endl;
      }
      os << "" << std::endl;
    }
    return os;
  }


  template <class T>
  class array3d_distribute : public array3d<T> {
    public:
      std::size_t glob_Rows, glob_Cols, glob_Heights;

      /* MPI Topology Features */
      int rank, num_proc;
      int nbr_up, nbr_down, nbr_right, nbr_left, nbr_front, nbr_back;

      constexpr static int dimension {3};

      int starts[dimension], ends[dimension], coordinates[dimension];

      MPI_Datatype vecs[dimension];
      MPI_Comm     communicator;


    public: 
      array3d_distribute() : glob_Rows{0}, glob_Cols{0}, glob_Heights{0}, array3d<T>(0,0,0) {}; 

      /**
       * @brief Generate the distributed array by inputting communicator and global sizes.
       * 
       * @param gRows Global number of rows.
       * @param gCols Global number of columns.
       * @param dims Array of dimensions.
       * @param comm MPI communicator.
      */
      void distribute(std::size_t gRows, std::size_t gCols, std::size_t gHeights, 
                        const int dims[3], MPI_Comm comm)
      {
        communicator = comm;
        glob_Rows = gRows; glob_Cols = gCols, glob_Heights = gHeights;

        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &num_proc);
        MPI_Cart_coords(comm, rank, dimension, coordinates);

        MPI_Cart_shift(comm, 0, 1, &nbr_up,   &nbr_down );
        MPI_Cart_shift(comm, 1, 1, &nbr_left, &nbr_right);
        MPI_Cart_shift(comm, 2, 1, &nbr_front, &nbr_back);

        Decomp1d(glob_Rows-2, dims[0], coordinates[0], starts[0], ends[0]);
        Decomp1d(glob_Cols-2, dims[1], coordinates[1], starts[1], ends[1]);
        Decomp1d(glob_Heights-2, dims[2], coordinates[2], starts[2], ends[2]);

        /* Setup vector types of halo */ 
        const int nx {ends[0] - starts[0] + 1 + 2};
        const int ny {ends[1] - starts[1] + 1 + 2};
        const int nz {ends[2] - starts[2] + 1 + 2};




        this->resize(nx, ny, nz);
      }


    // Update data
    private:
      double coff;
      double diag_x, diag_y, weight_x, weight_y;
      double dt, hx, hy;

    // Finite Difference Method (FDM)
    public:
      // Heat Equation
      void sweep_setup_heat3d(double coff, double time);
      void sweep_heat3d(array3d_distribute<T>&out);
      


  };



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