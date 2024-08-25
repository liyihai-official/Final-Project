#include <torch/torch.h>
#include <torch/script.h> // for torch::save

#include <numbers>
#include <random>
#include <vector>
#include <chrono>

#include <types.hpp>
#include <pinn/helper.hpp>

#include <pinn/pinn.hpp>
#include <pinn/types.hpp>
#include <pinn/dataset.hpp>

Integer 
  main( Integer argc, Char ** argv)
{
  constexpr maintype tol {1E-4};
  constexpr size_type nsteps {10000000};
  final_project::PINN::Dimension Dimension {final_project::PINN::Dimension::UNKNOWN};
  final_project::String filename {""};
  Integer opt {0}, iter {1};  

  while ((opt = getopt(argc, argv, "HhD:d:L:l:")) != -1)  // Command Line Arguments
  {
switch (opt)
{
  case 'L': case 'l':
    filename = optarg;
    std::cout 
      << "This program will trained model to the file: " 
      << filename << std::endl;
    break;
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

  torch::Device device { torch::cuda::is_available() ? torch::kCUDA : torch::kCPU };
  size_type numDim {0}, nx {NX}, ny {NY}, nz {NZ};
  final_project::PINN::dataset dataset;

  switch (Dimension)
  {
    case final_project::PINN::Dimension::PINN_2D:
      numDim = IN_SIZE_2D;
      dataset = final_project::PINN::dataset(nx, ny, DATASET_X_INTERNAL, DATASET_X_BOUNDARY, DATASET_Y_BOUNDARY, device);
      break;
    case final_project::PINN::Dimension::PINN_3D:
      numDim = IN_SIZE_3D;
      dataset = final_project::PINN::dataset(nx, ny, nz, DATASET_X_INTERNAL, DATASET_X_BOUNDARY, DATASET_Y_BOUNDARY, device);
      break;
    default:
      final_project::PINN::helper_message();
      FINAL_PROJECT_ASSERT(Dimension == final_project::PINN::Dimension::UNKNOWN);
      break;
  }


  auto net { final_project::PINN::HeatPINN(numDim, OUT_SIZE, /* hsize */ 10) };
  net->to(device);

  torch::optim::Adam adam_optim( net->parameters(), torch::optim::AdamOptions(1E-3) );
  torch::Tensor loss_sum;


  auto start = std::chrono::high_resolution_clock::now();
  while (iter <= nsteps)
  { 
    auto closure = [&](){
      adam_optim.zero_grad();
      
      loss_sum = final_project::PINN::get_total_loss(
        net, dataset.X_internal, dataset.X_boundary, dataset.Y_boundary, device);

      loss_sum.backward();

      return loss_sum;
    };

    adam_optim.step(closure);
    if (iter % 30 == 0)
    {
      std::cout 
        << "Iteration = "           << std::fixed << std::setw(8) << iter << "\t"
        << "Loss = " << std::fixed  << std::setprecision(8) << std::setw(12) << loss_sum.item<maintype>() << "\t"
        << "Loss.Device.Type = "    << loss_sum.device().type() 
        << std::endl; 
    }

    ++iter;
    if (loss_sum.item<maintype>() < tol)
    {
#ifndef NDEBUG
std::cout 
  << "Training stopped." << "\n"
  << "Final iter=" << iter - 1 << "\n"
  << "loss=" << std::setprecision(7) << loss_sum.item<maintype>()  << "\n"
  << "loss.device().type()=" << loss_sum.device().type() 
  << std::endl;
#endif // end NDEBUG
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
  if (!filename.empty())  torch::save(net, filename);


//   Integer iter {1};
//   constexpr size_type nsteps {1'000};
//   constexpr size_type numDim {2}, nx {NX}, ny {NY};

//   torch::Device device { torch::cuda::is_available() ? torch::kCUDA : torch::kCPU };

//   final_project::PINN::dataset dataset (nx, ny, DATASET_X_INTERNAL, DATASET_X_BOUNDARY, DATASET_Y_BOUNDARY, device);

//   std::cout << dataset.X_boundary << std::endl;
// #ifndef NDEBUG
//   // dataset.show_Y_boundary();
//   // dataset.show_X_internal();
//   // dataset.show_X_boundary();
// #endif

//   // Predefined Arguments
//   std::cout << "Problem size: "           << "\n"
//             << "\tRows: "       << nx-2   << "\n"
//             << "\tColumns: "    << ny-2       << std::endl;
//   std::cout << "Running ON: "   << device     << std::endl;

//   auto net { final_project::PINN::HeatPINN(IN_SIZE_2D, OUT_SIZE, /*hsize*/ 20) };
//   net->to(device);

//   torch::Tensor loss_sum;

//   torch::optim::Adam adam_optim( net->parameters(), torch::optim::AdamOptions(1E-3) );

//   std::cout << dataset.X_boundary.sizes() << std::endl;
//   std::cout << dataset.X_internal.sizes() << std::endl;
//   std::cout << dataset.Y_boundary.sizes() << std::endl;

//   while (iter <= nsteps)
//   { 
//     auto closure = [&](){
//       adam_optim.zero_grad();
//       loss_sum = final_project::PINN::get_total_loss(
//         net, dataset.X_internal, dataset.X_boundary, dataset.Y_boundary, device);
//       loss_sum.backward();
//       return loss_sum;
//     };

//     adam_optim.step(closure);

//     if (iter % 30 == 0)
//     {
//       std::cout 
//         << "Iteration = " << iter << "\t"
//         << "Loss = " << std::fixed << std::setprecision(4) << std::setw(8) << loss_sum.item<Float>() << "\t"
//         << "Loss.Device.Type = " << loss_sum.device().type() 
//         << std::endl; 
//     }

//     ++iter;
//     if (loss_sum.item<Float>() < 1E-4) break;
//   }

//   // std::cout 
//   //   << "Training stopped." << "\n"
//   //   << "Final iter=" << iter - 1 << "\n"
//   //   << "loss=" << std::setprecision(7) << loss_sum.item<Float>()  << "\n"
//   //   << "loss.device().type()=" << loss_sum.device().type() 
//   //   << std::endl;

  // Integer pNX {100}, pNY {100};
  // final_project::PINN::dataset valset (IN_SIZE_2D, OUT_SIZE, pNX, pNY, device);
  // auto out = net->forward(val_dataset.X_internal);

  // final_project::multi_array::array_base<Float, numDim> gather (pNX, pNY);

  // for (Integer x = 0; x < pNX; ++x)
  // {
  //   for (Integer y = 0; y < pNY; ++y)
  //   {
  //     gather(x,y) = out.index({x * pNY + y}).item<Float>();
  //   }
  // }

  // // std::cout << gather.data() << std::endl;
  // gather.saveToBinary("test.bin");

  // Integer pNX {20}, pNY {20},  pNZ {20};
  // final_project::PINN::dataset valset (3, OUT_SIZE, pNX, pNY, pNZ, device);
  // auto out = net->forward(valset.X_internal);

  // final_project::multi_array::array_base<maintype, 3> gather (pNX, pNY, pNZ);

  // for (Integer x = 0; x < pNX; ++x)
  //   for (Integer y = 0; y < pNY; ++y)
  //     for (Integer z = 0; z < pNZ; ++z)
  //       gather(x,y,z) = out.index({x * pNY * pNZ + y * pNZ + z}).item<maintype>();
  // // std::cout << gather.data() << std::endl;
  // gather.saveToBinary("test_3d.bin");


  return 0;

}