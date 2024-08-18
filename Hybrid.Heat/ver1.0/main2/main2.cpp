#include <torch/torch.h>
#include <torch/script.h> // for torch::save

#include <numbers>
#include <random>
#include <vector>

#include <types.hpp>

#include <pinn/pinn.hpp>
#include <pinn/dataset.hpp>

/// Problem Size + Boundaries
#if !defined(NX) || !defined (NY) || !defined(NZ)
#define NX 200+2
#define NY 200+2
#endif

#if !defined (IN_SIZE) || !defined (OUT_SIZE)
#define IN_SIZE 2
#define OUT_SIZE 1
#endif 

/// Using datatypes
using Integer = final_project::Integer;
using Char    = final_project::Char;
using Double  = final_project::Double;
using Float   = final_project::Float;
using size_type = final_project::Dworld;

using maintype  = Float;

using BCFunction = std::function<maintype(maintype, maintype)>;

int main ()
{
  Integer iter {1};
  constexpr size_type nsteps {1'000};
  constexpr size_type numDim {2}, nx {NX}, ny {NY};

  torch::DeviceType dp { torch::cuda::is_available() ? torch::kCUDA : torch::kCPU };
  torch::Device device { torch::cuda::is_available() ? torch::kCUDA : torch::kCPU };

  std::cout << "Problem size: " 
            << "\n\tRows: "     << nx-2 
            << "\n\tColumns: "  << ny-2       << std::endl;
  std::cout << "Running ON: "   << dp         << std::endl;


  std::mt19937 rde {std::random_device{}()};
  std::uniform_real_distribution<Float> rng(0.0, 1.0);

  BCFunction Dim00 {[](maintype x, maintype y){ return y;}};
  BCFunction Dim01 {[](maintype x, maintype y){ return 1;}};

  BCFunction Dim10 {[](maintype x, maintype y){ return x;}};
  BCFunction Dim11 {[](maintype x, maintype y){ return 1;}};


  final_project::PINN::dataset dataset (IN_SIZE, OUT_SIZE, NX, NY, rde, rng, Dim00, Dim01, Dim10, Dim11, device);

  // dataset.show_Y_boundary();
  // dataset.show_X_internal();
  // dataset.show_X_boundary();

  auto net { final_project::PINN::HeatPINN(IN_SIZE, OUT_SIZE, /*hsize*/ 20) };
  net->to(device);

  torch::Tensor loss_sum;

  torch::optim::Adam adam_optim(
    net->parameters(), 
    torch::optim::AdamOptions(1E-3)
  );

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

    // if (iter % 300 == 0)
    // {
    //   std::cout 
    //     << "Iteration = " << iter << "\t"
    //     << "Loss = " << std::fixed << std::setprecision(4) << std::setw(8) << loss_sum.item<Float>() << "\t"
    //     << "Loss.Device.Type = " << loss_sum.device().type() 
    //     << std::endl; 
    // }

    ++iter;
    if (loss_sum.item<Float>() < 1E-2) break;
  }



  // std::cout 
  //   << "Training stopped." << "\n"
  //   << "Final iter=" << iter - 1 << "\n"
  //   << "loss=" << std::setprecision(7) << loss_sum.item<Float>()  << "\n"
  //   << "loss.device().type()=" << loss_sum.device().type() 
  //   << std::endl;

  // torch::save(net, "model.pt");


  // final_project::PINN::dataset valset (IN_SIZE, OUT_SIZE, NX, NY, device);
  // auto out = net->forward(valset.X_internal);
  // std::cout << out << std::endl;

  // final_project::multi_array::array_base<Float, 2> gather (NX, NY);

  // for (Integer x = 0; x < NX; ++x)
  // {
  //   for (Integer y = 0; y < NY; ++y)
  //   {
  //     gather(x,y) = out.index({x * NY + y}).item<Float>();
  //   }
  // }

  // std::cout << gather.data() << std::endl;
  // gather.saveToBinary("test.bin");

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


