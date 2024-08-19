#include <torch/torch.h>
#include <torch/script.h> // for torch::save

#include <numbers>
#include <random>
#include <vector>

#include <types.hpp>

#include <pinn/pinn.hpp>
#include <pinn/types.hpp>
#include <pinn/dataset.hpp>


int main ()
{
  Integer iter {1};
  constexpr size_type nsteps {1'000};
  constexpr size_type numDim {2}, nx {NX}, ny {NY};

  torch::Device device { torch::cuda::is_available() ? torch::kCUDA : torch::kCPU };

  final_project::PINN::dataset dataset (nx, ny, DATASET_X_INTERNAL, DATASET_X_BOUNDARY, DATASET_Y_BOUNDARY, device);

#ifndef NDEBUG
  // dataset.show_Y_boundary();
  // dataset.show_X_internal();
  // dataset.show_X_boundary();
#endif

  // Predefined Arguments
  std::cout << "Problem size: "           << "\n"
            << "\tRows: "       << nx-2   << "\n"
            << "\tColumns: "    << ny-2       << std::endl;
  std::cout << "Running ON: "   << device     << std::endl;

  auto net { final_project::PINN::HeatPINN(IN_SIZE_2D, OUT_SIZE, /*hsize*/ 20) };
  net->to(device);



  torch::Tensor loss_sum;

  torch::optim::Adam adam_optim( net->parameters(), torch::optim::AdamOptions(1E-3) );

  std::cout << dataset.X_boundary.sizes() << std::endl;
  std::cout << dataset.X_internal.sizes() << std::endl;
  std::cout << dataset.Y_boundary.sizes() << std::endl;

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
        << "Iteration = " << iter << "\t"
        << "Loss = " << std::fixed << std::setprecision(4) << std::setw(8) << loss_sum.item<Float>() << "\t"
        << "Loss.Device.Type = " << loss_sum.device().type() 
        << std::endl; 
    }

    ++iter;
    if (loss_sum.item<Float>() < 1E-4) break;
  }



  // std::cout 
  //   << "Training stopped." << "\n"
  //   << "Final iter=" << iter - 1 << "\n"
  //   << "loss=" << std::setprecision(7) << loss_sum.item<Float>()  << "\n"
  //   << "loss.device().type()=" << loss_sum.device().type() 
  //   << std::endl;

  torch::save(net, "model.pt");

  Integer pNX {100}, pNY {100};
  final_project::PINN::dataset valset (IN_SIZE_2D, OUT_SIZE, pNX, pNY, device);
  auto out = net->forward(valset.X_internal);
  // std::cout << out << std::endl;

  final_project::multi_array::array_base<Float, 2> gather (pNX, pNY);

  for (Integer x = 0; x < pNX; ++x)
  {
    for (Integer y = 0; y < pNY; ++y)
    {
      gather(x,y) = out.index({x * pNY + y}).item<Float>();
    }
  }

  // std::cout << gather.data() << std::endl;
  gather.saveToBinary("test.bin");

  return 0;

}