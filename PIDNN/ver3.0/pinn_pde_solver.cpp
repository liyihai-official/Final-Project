#include <torch/torch.h>

#include <iostream>
#include <iomanip>

#include <random>
#include <vector>
#include <cassert>

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


struct HeatPINNImpl
  : torch::nn::Module
{
  HeatPINNImpl()                  = default;
  HeatPINNImpl(int ,int, int);
  torch::Tensor forward(torch::Tensor);

  torch::nn::Linear input, h0, output; 
};
TORCH_MODULE(HeatPINN);

void get_whole_dataset_X(
  std::vector<Float> &, 
  std::mt19937 & , 
  std::uniform_real_distribution<Float> &, 
  size_type, size_type);

void get_bc_dataset_X(
  std::vector<Float> &, 
  std::mt19937 & , 
  std::uniform_real_distribution<Float> &, 
  size_type, size_type);

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


/// ------------------ Y train ------------------ ///

torch::Tensor Y_train {
  torch::zeros({2 * (NX + NY, OUT_SIZE)}, device)
};

#ifndef NDEBUG
  std::cout 
    << "-------------------------------------- Y train -------------------------------------- "  << "\n"
    << "y_train sizes: "            << Y_train.sizes()        << "\n"
    << "y_train.device().type(): " << Y_train.device().type() << "\n"
    << "y_train.requires_grad(): " << Y_train.requires_grad() << "\n"
    << std::endl;
#endif

/// ------------------ GRID INTERNAL ------------------ ///
  std::vector<Float> X_data;
  get_whole_dataset_X(X_data, rde, rng, NX, NY);
  torch::Tensor X {
    torch::from_blob(
      X_data.data(), 
      {NX * NY, IN_SIZE}
    ).to(device)
  };

#ifndef NDEBUG // See the Initialized dataset in the internal grid 
  std::cout 
    << "----------------------------------- GRID INTERNAL -----------------------------------"  << "\n"
    << "X sizes (grid): "     << X.sizes()                    << "\n"
    << "X.device().index(): " << X.device().index()           << "\n"
    << "X.requires_grad(): "  << X.requires_grad()            << "\n"
    << std::endl;

  for (size_type x = 0; x < NX; ++x)
  {
    std::cout << std::fixed << std::setw(3) << x;
    for (size_type y = 0; y < 2 * NY; y+=2)
    {
      auto index {x * NY + y};
      std::cout 
        << std::fixed << std::setprecision(2) << std::setw(4)
        << "["  << X_data[index] 
        << ", " << X_data[index + 1]
        << "]  ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;


#endif 


/// ------------------ GRID BOUNDARY ------------------ ///
  std::vector<Float> X_train_data;
  get_bc_dataset_X(X_train_data, rde, rng, NX, NY);

  torch::Tensor X_train {
    torch::from_blob(
      X_train_data.data(),
      { 2 * (NX + NY), IN_SIZE }
    ).to(device)
  };

#ifndef NDEBUG

  std::cout 
    << "---------------------------------- GRID BOUNDARY -----------------------------------"  << "\n"
    << "X_train sizes (grid): "       << X_train.sizes()          << "\n"
    << "X_train.device().index(): "   << X_train.device().index() << "\n"
    << "X_train.requires_grad(): "    << X_train.requires_grad()  << "\n"
    << std::endl;

  for (size_type idx = 0; idx < 2 * 2 * (NX + NY); idx+=2)
  {
      std::cout 
        << std::fixed << std::setw(3) << idx / 2
        << std::fixed << std::setprecision(3) << std::setw(4)
        << "["  << X_train_data[idx] 
        << ", " << X_train_data[idx+1]
        << "]  "
        << std::endl;
  }
  std::cout << std::endl;

#endif 




#ifndef NDEBUG 
  std::cout 
    << "------------------------------------------"
    << " Training Process "
    << "------------------------------------------" << "\n"
    << std::endl;
#endif 

  auto net { HeatPINN(IN_SIZE, OUT_SIZE, 2) };
  net->to(device);

  torch::Tensor loss_sum;
  Integer iter {1};

  torch::optim::Adam adam_optim(
    net->parameters(), 
    torch::optim::AdamOptions(1E-3)
  );


  return 0;
}


void 
get_whole_dataset_X(std::vector<Float> & obj,
  std::mt19937 & rde, 
  std::uniform_real_distribution<Float> & rng,
  size_type nx, size_type ny)
{
  for (size_type x = 0; x < nx; ++x)
  {
    for (size_type y = 0; y < ny; ++y)
    {
      // (x, y) 
      obj.push_back(rng(rde)); 
      obj.push_back(rng(rde));
    }
  }
}

void get_bc_dataset_X(std::vector<Float> & obj,
  std::mt19937 & rde, 
  std::uniform_real_distribution<Float> & rng,
  size_type nx, size_type ny)
{

    for (size_type y = 0; y < ny; ++y)
    {
      obj.push_back(0.0);
      obj.push_back(rng(rde));

      obj.push_back(1.0);
      obj.push_back(rng(rde));
    }

    for (size_type x = 0; x < nx; ++x)
    {
      obj.push_back(rng(rde));
      obj.push_back(0.0);

      obj.push_back(rng(rde));
      obj.push_back(1.0);
    }

}


inline 
HeatPINNImpl::HeatPINNImpl(Integer insize, Integer outsize, Integer hidsize)
: 
  input(torch::nn::Linear(insize, hidsize)),
  h0(torch::nn::Linear(hidsize, hidsize)),
  output(torch::nn::Linear(hidsize, outsize))
{
  register_module("Input", input);
  register_module("hidden_0", h0);
  register_module("output", output);
}