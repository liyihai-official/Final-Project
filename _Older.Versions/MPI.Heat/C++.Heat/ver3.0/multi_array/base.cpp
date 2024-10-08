/**
 * @file base.cpp
 * @brief This file contains the definition of the array classes
 *        and features for parallel processing.
 * 
 * The classes provided in this file are designed to set up basic 
 * features of a skeleton 1D, 2D and 3D array. And enhanced them into
 * distributed 1D, 2D and 3D array used in parallel processing 
 * environments.
 * 
 * 
 * @author LI Yihai
 * @version 3.0
 * @date May 25, 2024
 */

#ifndef FINAL_PROJECT_MULTI_BASE_HPP_LIYIHAI
#define FINAL_PROJECT_MULTI_BASE_HPP_LIYIHAI

#pragma once
#include <iostream>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <iterator>
#include <unistd.h>


/// For random number generator
#include <functional>
#include <random>

#include <fstream>

#include "../types.cpp"
#include "../assert.cpp"

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

////////////////////////////////////////////////////////////////////////////////////////////////////////////
//          1D array
////////////////////////////////////////////////////////////////////////////////////////////////////////////


  /**
   * @brief A 1D array class.
   * 
   * @tparam T The type of the elements stored in the array.
  */
  template <class T>
  class array1d {
    public:
      typedef T               value_type;
      typedef T&              reference;
      typedef const T&        const_reference;
      typedef T*              iterator;
      typedef const T*        const_iterator;


      typedef std::size_t     size_type;
      typedef final_project::_detail::_size_type             _size_type;
      typedef final_project::_detail::_multi_array_shape<1>  _shape_type;

    public:
      array1d(size_type Num) 
        : N {Num}, data{std::make_unique<T[]>(Num)} {}
      // array1d(_size_type Num) 
      //   : _shape({Num}), data{std::make_unique<T[]>(Num)} {}

  
    public:
      size_type N;
      std::unique_ptr<T[]> data;

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

      // Resize
      void resize(size_type new_N)
      {
        this->N = new_N;
        this->data = std::make_unique<T[]>(new_N);  
      }

      // Friend function declarations
      template <class U>
      friend std::ostream& operator<<(std::ostream& os, const array1d<U>& in);
      
      void saveToBinaryFile(const std::string& filename) const;
  };



////////////////////////////////////////////////////////////////////////////////////////////////////////////
//          2D array
////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief A 2D array class.
   * 
   * @tparam T The type of the elements stored in the array.
   */
  template <class T>
  class array2d : public array1d<T> {

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
      size_type Rows, Cols;
      
    public:
      array2d()
        : Rows{0}, Cols{0}, array1d<T>(0) {}

      array2d(size_type Rows, size_type Cols) 
        : Rows{Rows}, Cols{Cols}, array1d<T>(Rows * Cols) {}

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

      template <typename T2>
      array2d<T> operator* (const array2d<T2> &rhs) const
      {
FINAL_PROJECT_ASSERT_MSG( (Cols == rhs.Rows), "Invalid Size of Multiplication.");
        auto result {array2d<T>(Rows, rhs.Cols)};

        for (size_type i = 0; i < Rows; ++i) 
          for (size_type j = 0; j < rhs.Cols; ++j)
          {
            result(i, j) = 0;
            for (size_type k = 0; k < Cols; ++k)
              result(i, j) += (*this)(i, k) * rhs(k, j);
          }
        return result;
      }

      template <typename T2>
      array2d<T> operator+ (const array2d<T2> & rhs) const 
      {
FINAL_PROJECT_ASSERT_MSG( (Rows == rhs.Rows && Cols == rhs.Cols), "Invalid Size of Multiplication.");
        auto result {array2d<T>(Rows, Cols)};
        for (size_type i = 0; i < this->size(); ++i) result(i) = rhs(i) + (*this)(i);
      }


      // assignment operator = 
      template <typename T2>
      array2d<T>& operator= (const array2d<T2> & rhs)
      {
        std::copy(rhs.begin(), rhs.end(), begin());
        return *this;
      }

      template <typename T2>
      array2d<T>& operator=(array2d<T2>&& other) noexcept 
      {
        if (this != &other) {
          Rows = other.Rows;
          Cols = other.Cols;
          this->data = std::move(other.data);

          other.Rows = 0;
          other.Cols = 0;
        }
        return *this;
    }
      

      // assign one value to all data
      void assign (const T& value) { fill (value); }
      void fill (const T& value)
      {
        std::fill_n(begin(), size(), value);
      }

      /// @brief  Fill the matrix by random values
      /// @attention The distribution of R.V is default as Uniform Distribution.
      void fill_random()
      {
        std::random_device rd{};
        std::default_random_engine eng{rd()};

        auto ui = std::bind( std::normal_distribution<>{}, std::ref(eng) );

        std::generate(begin(), end(), ui);
      }

      /// @brief Swap, Swap the input 2d matrix in to this matrix, which should have
      ///         identical shape.
      /// @param other The reference of matrix that will be Swapped.
      void swap (array2d<T>& other)
      {
        for (size_type i = 0; i < size(); ++i)
          this->data.swap(other.data);
      }


      /// @brief Resize the array.
      /// @param new_rows The new number of rows.
      /// @param new_cols The new number of columns.
      void resize(size_type new_rows, size_type new_cols) {
        Rows = new_rows;
        Cols = new_cols;
        this->data = std::make_unique<T[]>(new_rows * new_cols);
        this->N = new_rows * new_cols;
      }

      /// @brief Transpose the 2d array.
      /// @attention This method is unavoidable memory unfriendly. 
      ///           Here is the naive way for doing this, more advanced algorithms 
      ///           are needed when the matrices getting larger enough.
      void transpose()
      {
        auto other {array2d<T>(Cols, Rows)};

        for (size_type i = 0; i < Rows; ++i)
          for (size_type j = 0; j < Cols; ++j)
            other(j, i) = (*this)(i, j);
        
        swap(other);
        this->Rows = other.Cols;
        this->Cols = other.Rows;
      }

      // Friend function declarations
      template <class U>
      friend std::ostream& operator<<(std::ostream& os, const array2d<U>& in);

      // Save the array to a binary file
      /**
       * @brief save 2D array to Binary File
       * 
       * @details This version is aiming to maintain the continuity of program
       * aligned with 2D version.
       * 
       * @param std::string& filename The filename
      */
      void saveToBinaryFile(const std::string& filename) const;

  }; /* class array2d */


////////////////////////////////////////////////////////////////////////////////////////////////////////////
//          3D array
////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
      void saveToBinaryFile(const std::string& filename) const;
 
  };  /* class array3d */

} // namespace final_project





#endif /* FINAL_PROJECT_ARRAY_HPP */