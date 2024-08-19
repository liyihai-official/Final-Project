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
  constexpr size_type numDim {IN_SIZE_3D}, nx {NX}, ny {NY}, nz {NZ};

  torch::Device device { torch::cuda::is_available() ? torch::kCUDA : torch::kCPU };

  final_project::PINN::dataset_3d dataset (nx, ny, nz, DATASET_X_INTERNAL, DATASET_X_BOUNDARY, DATASET_Y_BOUNDARY, device);

// #ifndef NDEBUG
//   dataset.show_Y_boundary();
//   dataset.show_X_internal();
//   dataset.show_X_boundary();
// #endif

  // Predefined Arguments
  std::cout << "Problem size: "           << "\n"
            << "\tRows: "       << nx-2   << "\n"
            << "\tColumns: "    << ny-2   << "\n"
            << "\tDepths: "     << nz-2   << std::endl;
  std::cout << "Running ON: "   << device << std::endl;

  // auto net { final_project::PINN::HeatPINN(IN_SIZE_3D, OUT_SIZE, /*hsize*/ 20) };
  // net->to(device);



  // torch::Tensor loss_sum;

  // torch::optim::Adam adam_optim( net->parameters(), torch::optim::AdamOptions(1E-3) );


  // while (iter <= nsteps)
  // { 
  //   auto closure = [&](){
  //     adam_optim.zero_grad();
  //     loss_sum = final_project::PINN::get_total_loss(
  //       net, dataset.X_internal, dataset.X_boundary, dataset.Y_boundary, device);
  //     loss_sum.backward();
  //     return loss_sum;
  //   };

  //   adam_optim.step(closure);

  //   if (iter % 30 == 0)
  //   {
  //     std::cout 
  //       << "Iteration = " << iter << "\t"
  //       << "Loss = " << std::fixed << std::setprecision(4) << std::setw(8) << loss_sum.item<Float>() << "\t"
  //       << "Loss.Device.Type = " << loss_sum.device().type() 
  //       << std::endl; 
  //   }

  //   ++iter;
  //   if (loss_sum.item<Float>() < 1E-4) break;
  // }



  // std::cout 
  //   << "Training stopped." << "\n"
  //   << "Final iter=" << iter - 1 << "\n"
  //   << "loss=" << std::setprecision(7) << loss_sum.item<Float>()  << "\n"
  //   << "loss.device().type()=" << loss_sum.device().type() 
  //   << std::endl;

  // torch::save(net, "model.pt");

  // Integer pNX {100}, pNY {100},  pNZ {100};
  // final_project::PINN::dataset_3d valset (numDim, OUT_SIZE, pNX, pNY, pNZ, device);
  // auto out = net->forward(valset.X_internal);
  // // std::cout << out << std::endl;

  // final_project::multi_array::array_base<Float, numDim> gather (pNX, pNY, pNZ);

  // for (Integer x = 0; x < pNX; ++x)
  // {
  //   for (Integer y = 0; y < pNY; ++y)
  //   {
  //     for (Integer z = 0; z < pNZ; ++z)
  //     {
  //       gather(x,y,z) = out.index({x * pNY * pNZ + y * pNZ}).item<Float>();
  //     }
  //   }
  // }


  final_project::multi_array::array_base<Float, 2> in (dataset.X_boundary.size(0), numDim);
  final_project::multi_array::array_base<Float, 1> u  (dataset.X_boundary.size(0));



  for (auto i = 0; i < dataset.X_boundary.size(0); ++i)
  {
    in(i,0) = dataset.X_boundary.index({i, 0}).item<Float>();
    in(i,1) = dataset.X_boundary.index({i, 1}).item<Float>();
    in(i,2) = dataset.X_boundary.index({i, 2}).item<Float>();
    u(i)    = dataset.Y_boundary.index({i, 0}).item<Float>();
  }


  in.saveToBinary("X.bin");
  u.saveToBinary("u.bin");

  // std::cout << gather.data() << std::endl;
  // gather.saveToBinary("test_3d.bin");

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


