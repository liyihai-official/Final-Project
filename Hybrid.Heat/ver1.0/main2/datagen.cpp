///
/// @file datagen.cpp
/// @brief Dataset Generator, this program will generate all datasets 
///         on 2d and 3d spaces with specified Dirichlet boundary condition.
///
///
/// @author LI Yihai
///
///
#include <torch/torch.h>
#include <torch/script.h> // for torch::save

#include <random>

#include <pinn/types.hpp>
#include <pinn/dataset.hpp>

Integer 
  main(Integer argc, Char ** argv)
{
  torch::Device device { torch::cuda::is_available() ? torch::kCUDA : torch::kCPU };

  std::mt19937 rde {std::random_device{}()};
  std::uniform_real_distribution<Float> rng(0.0, 1.0);

  /// 2D Boundary Conditions
  BCFunction Dim00 {[](maintype x, maintype y){ return y;}};
  BCFunction Dim01 {[](maintype x, maintype y){ return 1;}};

  BCFunction Dim10 {[](maintype x, maintype y){ return x;}};
  BCFunction Dim11 {[](maintype x, maintype y){ return 1;}};

  final_project::PINN::dataset dataset (IN_SIZE_2D, OUT_SIZE, NX, NY, rde, rng, Dim00, Dim01, Dim10, Dim11, device);

  dataset.save(DATASET_X_INTERNAL, DATASET_X_BOUNDARY, DATASET_Y_BOUNDARY);


  /// 3D Boundary Conditions
  BCFunction_3d Dim000 {[](maintype x, maintype y, maintype z){ return y + z - 2 * y * z; }};
  BCFunction_3d Dim001 {[](maintype x, maintype y, maintype z){ return 1 - y - z + 2 * y * z; }};

  BCFunction_3d Dim010 {[](maintype x, maintype y, maintype z){ return x + z - 2 * x * z; }};
  BCFunction_3d Dim011 {[](maintype x, maintype y, maintype z){ return 1 - x - z + 2 * x * z; }};

  BCFunction_3d Dim100 {[](maintype x, maintype y, maintype z){ return x + y - 2 * x * y; }};
  BCFunction_3d Dim101 {[](maintype x, maintype y, maintype z){ return 1 - x - y + 2 * x * y; }};

  final_project::PINN::dataset_3d dataset_3d (IN_SIZE_3D, OUT_SIZE, NX, NY, NZ, rde, rng, 
    Dim000, Dim001, 
    Dim010, Dim011, 
    Dim100, Dim101,
    device);

  dataset_3d.save(DATASET_X_INTERNAL, DATASET_X_BOUNDARY, DATASET_Y_BOUNDARY);

  return 0;
}