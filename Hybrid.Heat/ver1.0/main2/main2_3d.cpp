#include <torch/torch.h>
#include <torch/script.h> // for torch::save

#include <numbers>
#include <random>
#include <vector>
#include <chrono>

#include <types.hpp>

#include <pinn/pinn.hpp>
#include <pinn/types.hpp>
#include <pinn/dataset.hpp>


int main ()
{
  Integer iter {1};
  final_project::String filename {""};        // default empty filename (not saving results).

  constexpr maintype tol {1E-3};
  constexpr size_type nsteps {10000};
  constexpr size_type numDim {IN_SIZE_3D}, nx {NX}, ny {NY}, nz {NZ};

  torch::Device device { torch::cuda::is_available() ? torch::kCUDA : torch::kCPU };

  final_project::PINN::dataset_3d dataset (nx, ny, nz, DATASET_X_INTERNAL, DATASET_X_BOUNDARY, DATASET_Y_BOUNDARY, device);

  // Predefined Arguments
  std::cout << "Problem size: "           << "\n"
            << "\tRows: "       << nx-2   << "\n"
            << "\tColumns: "    << ny-2   << "\n"
            << "\tDepths: "     << nz-2   << std::endl;
  std::cout << "Running ON: "   << device << std::endl;


  /// Network
  auto net { final_project::PINN::HeatPINN(IN_SIZE_3D, OUT_SIZE, /*hsize*/ 20) };
  net->to(device);

  torch::Tensor loss_sum;

  torch::optim::Adam adam_optim( net->parameters(), torch::optim::AdamOptions(1E-3) );

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
        << "Iteration = "           << iter << "\t"
        << "Loss = " << std::fixed  << std::setprecision(4) << std::setw(8) << loss_sum.item<maintype>() << "\t"
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
  torch::save(net, "../out/model_3d.pt");





  Integer pNX {20}, pNY {20},  pNZ {20};
  final_project::PINN::dataset_3d valset (numDim, OUT_SIZE, pNX, pNY, pNZ, device);
  auto out = net->forward(valset.X_internal);

  final_project::multi_array::array_base<maintype, numDim> gather (pNX, pNY, pNZ);

  for (Integer x = 0; x < pNX; ++x)
    for (Integer y = 0; y < pNY; ++y)
      for (Integer z = 0; z < pNZ; ++z)
        gather(x,y,z) = out.index({x * pNY * pNZ + y * pNZ + z}).item<maintype>();
  // std::cout << gather.data() << std::endl;
  gather.saveToBinary("test_3d.bin");


  return 0;

}