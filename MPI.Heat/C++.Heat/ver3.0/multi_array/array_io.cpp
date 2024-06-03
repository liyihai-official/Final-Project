/**
 * @file array_io.cpp
 * @brief This file contains the definition of the array classes
 *        and I/O features.
 * 
 * The classes provided in this file are designed to set up I/O
 * features of a skeleton 1D, 2D and 3D array and in enhanced MPI 
 * envrionment.
 * 
 * 
 * @author LI Yihai
 * @version 3.0
 * @date May 25, 2024
 */
#ifndef FINAL_PROJECT_ARRAY_IO_HPP_LIYIHAI
#define FINAL_PROJECT_ARRAY_IO_HPP_LIYIHAI

#pragma once
#include <iostream>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <iterator>
#include <unistd.h>

#include <mpi.h>
#include <fstream>

#include "../assert.cpp"
#include "array.cpp"

namespace final_project 
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////
//          I/O
////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief I/O of array 1d.
   * 
   * @tparam T The type of elements stored in the array.
   * @param os The output stream.
   * @param in The array to output.
   * @return std::ostream& The output stream.
  */
  template <class T>
  std::ostream& operator<<(std::ostream& os, const array1d<T>& in) {
    for (std::size_t idx = 0; idx < in.N; ++idx) {
      os << "";
      os << std::fixed << std::setprecision(5) << std::setw(9) << in(idx);
      os << "";
    }
    std::cout << std::endl;
    return os;
  }

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
   * @brief I/O of array 3d.
   * 
   * @tparam T The type of elements stored in the array.
   * @param os The output stream.
   * @param in The array to output.
   * @return std::ostream& The output stream.
   */
  template <class T>
  std::ostream& operator<<(std::ostream& os, const array3d<T>& in) {
    for (std::size_t ridx = 0; ridx < in.Rows; ++ridx) 
    {
      os << "";
      for (std::size_t cidx = 0; cidx < in.Cols; ++cidx) 
      {
        os << "";
        for (std::size_t kidx = 0; kidx < in.Height; ++kidx) 
        {
          os << std::fixed << std::setprecision(5) << std::setw(9) << in(ridx, cidx, kidx);
        }
        os << "" << std::endl;
      }
      os << "" << std::endl;
    }
    return os;
  }

  // Save the array to a binary file
  /**
   * @brief save 1D array to Binary File
   * 
   * @details This version is aiming to maintain the continuity of program
   * aligned with 1D version.
   * 
   * @param std::string& filename The filename
  */
  template <class T>
  void array1d<T>::saveToBinaryFile(const std::string& filename) const {
      std::ofstream ofs(filename, std::ios::binary);
      if (!ofs) {
          throw std::runtime_error("Cannot open file");
      }
      ofs.write(reinterpret_cast<const char*>(&N), sizeof(N));
      ofs.write(reinterpret_cast<const char*>(this->data.get()), this->size() * sizeof(T));
  }

  /**
   * @brief save 2D array to Binary File
   * 
   * @details This version is aiming to maintain the continuity of program
   * aligned with 2D version.
   * 
   * @param std::string& filename The filename
  */
  template <class T>
  void array2d<T>::saveToBinaryFile(const std::string& filename) const {
      std::ofstream ofs(filename, std::ios::binary);
      if (!ofs) {
          throw std::runtime_error("Cannot open file");
      }
      ofs.write(reinterpret_cast<const char*>(&Rows), sizeof(Rows));
      ofs.write(reinterpret_cast<const char*>(&Cols), sizeof(Cols));
      ofs.write(reinterpret_cast<const char*>(this->data.get()), this->size() * sizeof(T));
  }

  // Save the array to a binary file
  /**
   * @brief save 3D array to Binary File
   * 
   * @details This version is aiming to solve the problem of 
   * MATLAB does not support ASCII format in saving 3D matrix
   * 
   * @param std::string& filename The filename
  */
  template <class T>
  void array3d<T>::saveToBinaryFile(const std::string& filename) const
  {
      std::ofstream ofs(filename, std::ios::binary);
      if (!ofs) {
          throw std::runtime_error("Cannot open file");
      }
      ofs.write(reinterpret_cast<const char*>(&Rows), sizeof(Rows));
      ofs.write(reinterpret_cast<const char*>(&Cols), sizeof(Cols));
      ofs.write(reinterpret_cast<const char*>(&Height), sizeof(Height));
      ofs.write(reinterpret_cast<const char*>(this->data.get()), this->size() * sizeof(T));
  }
  
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//          MPI-I/O
////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  /**
   * @brief Print the 1D array in order according to the rank of each processor.
   * 
   * This function ensures that each processor prints its portion of the 1D array in order,
   * based on their rank. It uses MPI barriers to synchronize the printing process and 
   * small sleeps to allow ordered output.
   * 
   * @tparam T The type of the elements in the array.
   * @param in The distributed 1D array to be printed.
   */
  template <class T>
  void print_in_order(final_project::array1d_distribute<T>& in)
  {
    MPI_Barrier(in.communicator);
    std::cout << "Attempting to print 1d array in order" << std::endl;
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


  /**
   * @brief Print the 3D array in order according to the rank of each processor.
   * 
   * This function ensures that each processor prints its portion of the 3D array in order,
   * based on their rank. It uses MPI barriers to synchronize the printing process and 
   * small sleeps to allow ordered output.
   * 
   * @tparam T The type of the elements in the array.
   * @param in The distributed 3D array to be printed.
   */
  template <class T>
  void print_in_order(final_project::array3d_distribute<T>& in)
  {
    MPI_Barrier(in.communicator);
    std::cout << "Attempting to print 3d array in order" << std::endl;
    sleep(0.01);
    MPI_Barrier(in.communicator);

    for ( int i = 0; i < in.num_proc; ++i)
    {
      if ( i == in.rank )
      {
        std::cout << "proc : " << in.rank << " at " 
        << "( "<< in.coordinates[0] << ", "<< in.coordinates[1] << ", " << in.coordinates[2] << " )"
        << "\n" << in << std::endl;
      }
      fflush(stdout);
      sleep(0.01);
      MPI_Barrier(in.communicator);
    }
  }

} // namespace final_project

#endif // end of FINAL_PROJECT_ARRAY_IO_HPP_LIYIHAI