#include <torch/torch.h>
#include <torch/script.h> // for torch::save



#include <random>
#include <vector>
#include <cassert>

#include <pinn/pinn.hpp>
#include <pinn/dataset.hpp>

using Integer = int;
using Float   = float;

using size_type = uint32_t;


#if !defined (NX) || !defined (NY)
#define NX 13
#define NY 9
#endif 

#if !defined (IN_SIZE) || !defined (OUT_SIZE)
#define IN_SIZE 2
#define OUT_SIZE 1
#endif 



// void get_whole_dataset_X(
//   std::vector<Float> &, 
//   std::mt19937 &, std::uniform_real_distribution<Float> &, 
//   size_type, size_type);

// void get_bc_dataset_X(
//   std::vector<Float> &, 
//   std::mt19937 & ,std::uniform_real_distribution<Float> &, 
//   size_type, size_type);


int main ()
{

  std::cout 
    << "---" 
    << " C++ Libtorh Example for Heat Equation 2D PINN"
    << "\n"          
    << std::endl;

  torch::DeviceType dp {
    torch::cuda::is_available() ? torch::kCUDA : torch::kCPU
  };
  torch::Device device(dp);

  std::cout << "Running ON: " << dp << std::endl;


  std::cout
    << "---------------------"
    << " Create Data Samples   // torch::Tensor X_train, Y_train, X; // "
    << "---------------------"
  << std::endl;
  std::mt19937 rde {std::random_device{}()};
  std::uniform_real_distribution<Float> rng(0.0, 1.0);

  final_project::PINN::dataset dataset (IN_SIZE, OUT_SIZE, NX, NY, rde, rng, device);
  dataset.show_Y_boundary();
  dataset.show_X_internal();
  dataset.show_X_boundary();

  auto net { final_project::PINN::HeatPINN(IN_SIZE, OUT_SIZE, /*hsize*/ 2) };
  net->to(device);
// /// ------------------ Y train ------------------ ///

// torch::Tensor Y_train {
//   torch::zeros({2 * (NX + NY),  OUT_SIZE}, device)
// };



// /// ------------------ GRID INTERNAL ------------------ ///
//   std::vector<Float> X_data;
//   get_whole_dataset_X(X_data, rde, rng, NX, NY);
//   torch::Tensor X {
//     torch::from_blob(
//       X_data.data(), 
//       {NX * NY, IN_SIZE},
//       torch::requires_grad()
//     ).to(device)
//   };

// #ifndef NDEBUG // See the Initialized dataset in the internal grid 

// #endif 


// /// ------------------ GRID BOUNDARY ------------------ ///
//   std::vector<Float> X_train_data;
//   get_bc_dataset_X(X_train_data, rde, rng, NX, NY);

//   torch::Tensor X_train {
//     torch::from_blob(
//       X_train_data.data(),
//       { 2 * (NX + NY), IN_SIZE }
//     ).to(device)
//   };

// #ifndef NDEBUG


// #endif 




// #ifndef NDEBUG 
//   std::cout 
//     << "------------------------------------------"
//     << " Training Process "
//     << "------------------------------------------" << "\n"
//     << std::endl;
// #endif 

  // auto net { final_project::PINN::HeatPINN(IN_SIZE, OUT_SIZE, 2) };
  // net->to(device);

  torch::Tensor loss_sum;
  Integer iter {1}, nsteps {10'000};

  torch::optim::Adam adam_optim(
    net->parameters(), 
    torch::optim::AdamOptions(1E-3)
  );


  while (iter <= nsteps)
  { 
    auto closure = [&](){
      loss_sum = final_project::PINN::get_total_loss(net, dataset.X_internal, dataset.X_boundary, dataset.Y_boundary, device);
      loss_sum.backward();
      return loss_sum;
    };

    adam_optim.step(closure);

    if (iter % 10 == 0)
    {
      std::cout 
        << "Iteration = " << iter << "\t"
        << "Loss = " << std::fixed << std::setprecision(4) << std::setw(8) << loss_sum.item<Float>() << "\t"
        << "Loss.Device.Type = " << loss_sum.device().type() 
        << std::endl; 
    }


    ++iter;
    if (loss_sum.item<Float>() < 1E-3)
    {
      break;
    }
  }



  std::cout 
    << "Training stopped." << "\n"
    << "Final iter=" << iter - 1 << "\n"
    << "loss=" << std::setprecision(7) << loss_sum.item<Float>()  << "\n"
    << "loss.device().type()=" << loss_sum.device().type() 
    << std::endl;

  torch::save(net, "model.pt");


  final_project::PINN::dataset valset (IN_SIZE, OUT_SIZE, NX, NY, device);
  auto out = net->forward(valset.X_internal);
  std::cout << out << std::endl;

  return 0;

}














// /// @brief 
// /// @param obj 
// /// @param rde 
// /// @param rng 
// /// @param nx 
// /// @param ny 
// void 
// get_whole_dataset_X(std::vector<Float> & obj,
//   std::mt19937 & rde, 
//   std::uniform_real_distribution<Float> & rng,
//   size_type nx, size_type ny)
// {
//   for (size_type x = 0; x < nx; ++x)
//   {
//     for (size_type y = 0; y < ny; ++y)
//     {
//       // (x, y) 
//       obj.push_back(rng(rde)); 
//       obj.push_back(rng(rde));
//     }
//   }
// }

// /// @brief 
// /// @param obj 
// /// @param rde 
// /// @param rng 
// /// @param nx 
// /// @param ny 
// void get_bc_dataset_X(std::vector<Float> & obj,
//   std::mt19937 & rde, 
//   std::uniform_real_distribution<Float> & rng,
//   size_type nx, size_type ny)
// {

//     for (size_type y = 0; y < ny; ++y)
//     {
//       obj.push_back(0.0);
//       obj.push_back(rng(rde));

//       obj.push_back(1.0);
//       obj.push_back(rng(rde));
//     }

//     for (size_type x = 0; x < nx; ++x)
//     {
//       obj.push_back(rng(rde));
//       obj.push_back(0.0);

//       obj.push_back(rng(rde));
//       obj.push_back(1.0);
//     }

// }








///
///
/// ====================================== Definitions about PDEs ====================================== ///
///
///


