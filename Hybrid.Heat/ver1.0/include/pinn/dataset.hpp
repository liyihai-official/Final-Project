///
/// @file dataset.hpp
/// @brief Generate Random dataset for training.
///


#ifndef FINAL_PROJECT_DATASET_HPP
#define FINAL_PROJECT_DATASET_HPP

#pragma once 
#include <torch/torch.h>

///
#include <random>
#include <types.hpp>
#include <pinn/types.hpp>
#include <multiarray.hpp>

///

#include <filesystem>


namespace final_project { namespace PINN {

/// @brief 
struct dataset
{
  
using BCFunction = std::function<value_type(value_type, value_type)>;
using BCFunction3D = std::function<value_type(value_type, value_type, value_type)>;

  dataset() = default;

  dataset& operator=(dataset &&)                noexcept;
  dataset& operator=(const dataset &)           = delete;


  /// @brief  Constructor of dataset (2d), load from predefined files and send them to given device.
  /// @param  NX grid size of data.
  /// @param  NY grid size of data.
  /// @param  dataset_X_internal Internal data needed to train.
  /// @param  dataset_X_boundary Boundary data contains information of grid points.
  /// @param  dataset_Y_boundary Boundary data of determinant values on grid points.
  /// @param  device Torch::Device, kCUDA or kCPU.
  dataset(const Integer, const Integer,
          const String, const String, const String,  const torch::Device &);

  /// @brief  Random generate datasets with given parameters.
  /// @param  IN_SIZE Dimension of Input layer of Torch::nn network.
  /// @param  OUT_SIZE Dimension of Output layer of Torch::nn network.
  /// @param  NX grid size of data.
  /// @param  NY grid size of data.
  /// @param  RNG Random number generator.
  /// @param  RE  Random number's distribution.
  /// @param  Dim00 Condition Function, on Dimension 0 and source site.
  /// @param  Dim01 Condition Function, on Dimension 0 and dest site.
  /// @param  Dim10 Condition Function, on Dimension 1 and source site.
  /// @param  Dim11 Condition Function, on Dimension 1 and dest site.
  /// @param  device Torch::Device, kCUDA or kCPU.
  dataset(const Integer /*IN_SIZE*/,  const Integer /*OUT_SIZE*/, 
          const Integer /*NX*/,       const Integer /*NY*/, 
          std::mt19937 & /* Random Number Generator */,
          std::uniform_real_distribution<value_type> & /* Distribution */,
          BCFunction &, BCFunction &, BCFunction &, BCFunction &, /* Boundary Functions */
          const torch::Device &);

  /// @brief  Generate a fined grid, rather than randomly.
  /// @param  IN_SIZE Dimension of Input layer of Torch::nn network.
  /// @param  OUT_SIZE Dimension of Output layer of Torch::nn network.
  /// @param  NX grid size of data.
  /// @param  NY grid size of data.
  /// @param  device Torch::Device, kCUDA or kCPU.
  dataset(const Integer /*IN_SIZE*/,  const Integer /*OUT_SIZE*/,
          const Integer /*NX*/,       const Integer /*NY*/,
          const torch::Device &);


  /// @brief  Constructor of dataset (3d), load from predefined files and send them to given device.
  /// @param  NX grid size of data.
  /// @param  NY grid size of data.
  /// @param  NZ grid size of data.
  /// @param  dataset_X_internal Internal data needed to train.
  /// @param  dataset_X_boundary Boundary data contains information of grid points.
  /// @param  dataset_Y_boundary Boundary data of determinant values on grid points.
  /// @param  device Torch::Device, kCUDA or kCPU.
  dataset( const Integer, const Integer, const Integer,
           const String, const String, const String,  const torch::Device &);

  /// @brief  Random generate datasets with given parameters.
  /// @param  IN_SIZE Dimension of Input layer of Torch::nn network.
  /// @param  OUT_SIZE Dimension of Output layer of Torch::nn network.
  /// @param  NX grid size of data.
  /// @param  NY grid size of data.
  /// @param  NZ grid size of data.
  /// @param  RNG Random number generator.
  /// @param  RE  Random number's distribution.
  /// @param  Dim00 Condition Function, on Dimension 0 and source site.
  /// @param  Dim01 Condition Function, on Dimension 0 and dest site.
  /// @param  Dim10 Condition Function, on Dimension 1 and source site.
  /// @param  Dim11 Condition Function, on Dimension 1 and dest site.
  /// @param  Dim20 Condition Function, on Dimension 2 and source site.
  /// @param  Dim21 Condition Function, on Dimension 2 and dest site.
  /// @param  device Torch::Device, kCUDA or kCPU.
  dataset( const Integer /*IN_SIZE*/,  const Integer /*OUT_SIZE*/, 
           const Integer /*NX*/,       const Integer /*NY*/,     const Integer /*NZ*/, 
           std::mt19937 & /* Random Number Generator */,
           std::uniform_real_distribution<value_type> & /* Distribution */,
           BCFunction3D &, BCFunction3D &, /* Boundary Functions */
           BCFunction3D &, BCFunction3D &, 
           BCFunction3D &, BCFunction3D &, 
           const torch::Device &);

  /// @brief  Generate a fined grid, rather than randomly.
  /// @param  IN_SIZE Dimension of Input layer of Torch::nn network.
  /// @param  OUT_SIZE Dimension of Output layer of Torch::nn network.
  /// @param  NX grid size of data.
  /// @param  NY grid size of data.
  /// @param  NZ grid size of data.
  /// @param  device Torch::Device, kCUDA or kCPU.
  dataset( const Integer /*IN_SIZE*/,  const Integer /*OUT_SIZE*/,
           const Integer /*NX*/,       const Integer /*NY*/,       const Integer /*NZ*/,
           const torch::Device &);

  /// @brief Save the datasets into files.
  /// @param  dataset_X_internal File name.
  /// @param  dataset_X_boundary File name.
  /// @param  dataset_Y_boundary File name.
  void save(const String, const String, const String);

  torch::Tensor X_internal, X_boundary, Y_boundary;
  Integer in_size, out_size, nx, ny, nz;
}; // dataset


// /// @brief 
// struct dataset_3d 
// {
//   using BCFunction = std::function<value_type(value_type, value_type, value_type)>;

//   dataset_3d() = default;

//   dataset_3d& operator=(dataset_3d &&)                noexcept;
//   dataset_3d& operator=(const dataset_3d &)           = delete;

//   /// @brief  Constructor of dataset (3d), load from predefined files and send them to given device.
//   /// @param  NX grid size of data.
//   /// @param  NY grid size of data.
//   /// @param  NZ grid size of data.
//   /// @param  dataset_X_internal Internal data needed to train.
//   /// @param  dataset_X_boundary Boundary data contains information of grid points.
//   /// @param  dataset_Y_boundary Boundary data of determinant values on grid points.
//   /// @param  device Torch::Device, kCUDA or kCPU.
//   dataset_3d( const Integer, const Integer, const Integer,
//               const String, const String, const String,  const torch::Device &);

//   /// @brief  Random generate datasets with given parameters.
//   /// @param  IN_SIZE Dimension of Input layer of Torch::nn network.
//   /// @param  OUT_SIZE Dimension of Output layer of Torch::nn network.
//   /// @param  NX grid size of data.
//   /// @param  NY grid size of data.
//   /// @param  NZ grid size of data.
//   /// @param  RNG Random number generator.
//   /// @param  RE  Random number's distribution.
//   /// @param  Dim00 Condition Function, on Dimension 0 and source site.
//   /// @param  Dim01 Condition Function, on Dimension 0 and dest site.
//   /// @param  Dim10 Condition Function, on Dimension 1 and source site.
//   /// @param  Dim11 Condition Function, on Dimension 1 and dest site.
//   /// @param  Dim20 Condition Function, on Dimension 2 and source site.
//   /// @param  Dim21 Condition Function, on Dimension 2 and dest site.
//   /// @param  device Torch::Device, kCUDA or kCPU.
//   dataset_3d( const Integer /*IN_SIZE*/,  const Integer /*OUT_SIZE*/, 
//               const Integer /*NX*/,       const Integer /*NY*/,     const Integer /*NZ*/, 
//               std::mt19937 & /* Random Number Generator */,
//               std::uniform_real_distribution<value_type> & /* Distribution */,
//               BCFunction &, BCFunction &, /* Boundary Functions */
//               BCFunction &, BCFunction &, 
//               BCFunction &, BCFunction &, 
//               const torch::Device &);

//   /// @brief  Generate a fined grid, rather than randomly.
//   /// @param  IN_SIZE Dimension of Input layer of Torch::nn network.
//   /// @param  OUT_SIZE Dimension of Output layer of Torch::nn network.
//   /// @param  NX grid size of data.
//   /// @param  NY grid size of data.
//   /// @param  NZ grid size of data.
//   /// @param  device Torch::Device, kCUDA or kCPU.
//   dataset_3d( const Integer /*IN_SIZE*/,  const Integer /*OUT_SIZE*/,
//               const Integer /*NX*/,       const Integer /*NY*/,       const Integer /*NZ*/,
//               const torch::Device &);

//   /// @brief Save the datasets into files.
//   /// @param  dataset_X_internal File name.
//   /// @param  dataset_X_boundary File name.
//   /// @param  dataset_Y_boundary File name.
//   void save(const String, const String, const String);

//   torch::Tensor X_internal, X_boundary, Y_boundary;
//   Integer in_size, out_size, nx, ny, nz;
// }; // dataset_3d

} // namespace PINN
} // namespace final_project  



#endif // end define FINAL_PROJECT_DATASET_HPP