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

#include <pinn/helper.hpp>
#include <pinn/types.hpp>
#include <pinn/dataset.hpp>

Integer 
  main(Integer argc, Char ** argv)
{
  final_project::PINN::Dimension Dimension {final_project::PINN::Dimension::UNKNOWN};
    Integer opt {0}, iter {1};
  while ((opt = getopt(argc, argv, "HhD:d:")) != -1)  // Command Line Arguments
  {
switch (opt)
{
  case 'H': case 'h':
    final_project::PINN::helper_message();
    exit(EXIT_FAILURE);
  case 'D': case 'd':
    Dimension = final_project::PINN::getDimensionfromString(optarg);
    break;
  default:
    std::cerr << "Invalid option: -" 
              << static_cast<Char>(opt)
              << "\n";
    exit(EXIT_FAILURE);   
}
  }

  final_project::PINN::dataset dataset;
  torch::Device device { torch::cuda::is_available() ? torch::kCUDA : torch::kCPU };
  std::mt19937 rde {std::random_device{42}()};
  std::uniform_real_distribution<Float> rng(0.0, 1.0);

  /// 2D Boundary Conditions
  BCFunction Dim00 {[](maintype x, maintype y){ return y;}};
  BCFunction Dim01 {[](maintype x, maintype y){ return 1;}};

  BCFunction Dim10 {[](maintype x, maintype y){ return x;}};
  BCFunction Dim11 {[](maintype x, maintype y){ return 1;}};

  /// 3D Boundary Conditions
  BCFunction_3d Dim000 {[](maintype x, maintype y, maintype z){ return y + z - 2 * y * z; }};
  BCFunction_3d Dim001 {[](maintype x, maintype y, maintype z){ return 1 - y - z + 2 * y * z; }};

  BCFunction_3d Dim010 {[](maintype x, maintype y, maintype z){ return x + z - 2 * x * z; }};
  BCFunction_3d Dim011 {[](maintype x, maintype y, maintype z){ return 1 - x - z + 2 * x * z; }};

  BCFunction_3d Dim100 {[](maintype x, maintype y, maintype z){ return x + y - 2 * x * y; }};
  BCFunction_3d Dim101 {[](maintype x, maintype y, maintype z){ return 1 - x - y + 2 * x * y; }};

  switch (Dimension)
  {
    case final_project::PINN::Dimension::PINN_2D:
      dataset = final_project::PINN::dataset(IN_SIZE_2D, OUT_SIZE, NX, NY,     rde, rng, 
        Dim00, Dim01, 
        Dim10, Dim11, 
        device);
      break;
    case final_project::PINN::Dimension::PINN_3D:
      dataset = final_project::PINN::dataset(IN_SIZE_3D, OUT_SIZE, NX, NY, NZ, rde, rng, 
        Dim000, Dim001, 
        Dim010, Dim011, 
        Dim100, Dim101,
        device);

      break;
    default:
      final_project::PINN::helper_message();
      FINAL_PROJECT_ASSERT(Dimension == final_project::PINN::Dimension::UNKNOWN);
      break;
  }

  dataset.save(DATASET_X_INTERNAL, DATASET_X_BOUNDARY, DATASET_Y_BOUNDARY);

  return 0;
}