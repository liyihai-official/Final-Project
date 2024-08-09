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


struct HeatPINN
  : torch::nn::Module
{
  HeatPINN()                  = default;
  HeatPINN(int ,int, int);
  torch::Tensor forward(torch::Tensor);

  torch::nn::Linear input;//, h0, h1, h2, h3, output; 
};

void get_whole_dataset_X(
  std::vector<Float> &, 
  std::default_random_engine & , 
  std::uniform_real_distribution<Float> &, 
  size_type, size_type);

void get_bc_dataset_X(
  std::vector<Float> &, 
  std::default_random_engine & , 
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
    << "---"
    << " Create Data Samples"
  << std::endl;
  std::default_random_engine rde {std::random_device{}()};
  std::uniform_real_distribution<Float> rng(0.0, 1.0);

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
  std::cout << "X sizes (grid): " << X.sizes()  << "\n"
            << "X.device().index(): " << X.device().index() << "\n"
            << "X.requires_grad(): " << X.requires_grad() << "\n"
            << std::endl;
  for (size_type x = 0; x < NX; ++x)
  {
    std::cout << std::fixed << std::setw(3) << x;
    for (size_type y = 0; y < NY; ++y)
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

  std::vector<Float> X_train_data;
  get_bc_dataset_X(X_train_data, rde, rng, NX, NY);



  // torch::Tensor X_train, Y_train, X;

  // Y_train {
  //   torch::zeros({})
  // }

  return 0;
}


void 
get_whole_dataset_X(std::vector<Float> & obj,
  std::default_random_engine & rde, 
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
  std::default_random_engine & rde, 
  std::uniform_real_distribution<Float> & rng,
  size_type nx, size_type ny)
{
  
}


inline 
HeatPINN::HeatPINN(int insize, int outsize, int hidsize)
: input(torch::nn::Linear(insize, hidsize))
{}