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
  final_project::PINN::Dimension Dimension {final_project::PINN::Dimension::UNKNOWN};
  final_project::PINN::dataset val_dataset;
  Integer pNX {100}, pNY {50}, pNZ {100};
  Integer opt {0};
  final_project::String filename {""}, SaveToFile {"result.bin"};

  torch::Device device { torch::cuda::is_available() ? torch::kCUDA : torch::kCPU };

  while ((opt = getopt(argc, argv, "HhD:d:L:l:F:f:")) != -1)  // Command Line Arguments
  {
switch (opt)
{
  case 'F': case 'f':
    SaveToFile = optarg;
    std::cout 
      << "The predictions are stored to the file: " 
      << SaveToFile << std::endl;
    break;
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

  if (!filename.empty()) 
  {
    switch (Dimension)
    {
      case final_project::PINN::Dimension::PINN_2D:
      {
        val_dataset = final_project::PINN::dataset(IN_SIZE_2D, OUT_SIZE, pNX, pNY, device);
        auto trained_net { final_project::PINN::HeatPINN(IN_SIZE_2D, OUT_SIZE, /* hsize */ 10) };
        torch::load(trained_net, filename); 
        trained_net->to(device);
        auto out = trained_net->forward(val_dataset.X_internal).to(torch::kCPU);;
        final_project::multi_array::array_base<maintype, IN_SIZE_2D> gather (pNX, pNY);

        gather.fill(out.data_ptr<maintype>(), out.data_ptr<maintype>() + out.numel());
        // std::cout << gather << std::endl;
        gather.saveToBinary(SaveToFile);

        break;
      }
      case final_project::PINN::Dimension::PINN_3D:
      {
        val_dataset = final_project::PINN::dataset(IN_SIZE_3D, OUT_SIZE, pNX, pNY, pNZ, device);
        auto trained_net { final_project::PINN::HeatPINN(IN_SIZE_3D, OUT_SIZE, /* hsize */ 10) };
        torch::load(trained_net, filename); 
        trained_net->to(device);
        auto out = trained_net->forward(val_dataset.X_internal).to(torch::kCPU);
        final_project::multi_array::array_base<maintype, IN_SIZE_3D> gather (pNX, pNY, pNZ);

        gather.fill(out.data_ptr<maintype>(), out.data_ptr<maintype>() + out.numel());
        // std::cout << gather << std::endl;
        gather.saveToBinary(SaveToFile);
        break;
      }
      default:
        final_project::PINN::helper_message();
        FINAL_PROJECT_ASSERT(Dimension == final_project::PINN::Dimension::UNKNOWN);
        break;
    }
  }

  return 0;

}