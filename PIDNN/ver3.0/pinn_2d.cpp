#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <random>

#include <vector>

/// @brief This is the module of the customized neural network
struct Net : torch::nn::Module
{

  torch::nn::Linear fc1, fc2, fc3;

  Net()
    : fc1(2 , 10),
      fc2(10, 10),
      fc3(10, 1)
    {
      register_module("fc1", fc1);
      register_module("fc2", fc2);
      register_module("fc3", fc3);
    }


  torch::Tensor forward(torch::Tensor x)
  {
    x = torch::tanh(fc1->forward(x));
    x = torch::tanh(fc2->forward(x));

    x = fc3->forward(x);

    return x;
  }

}; // end struct Net


int main ()
{
  // --------------------------------- CUDA device --------------------------------- //
  // begin of the device type 
  torch::DeviceType dp {  torch::cuda::is_available() ? 
                          torch::kCUDA : torch::kCPU };
  torch::Device device(dp);
  // end of get the device type;

  // --------------------- Begin of generating random datasets --------------------- //
  constexpr int n_bc {4}, n_data_per_bc {8}, n_c {10};

  std::default_random_engine rd {std::random_device{}()};
  std::uniform_real_distribution<float> rng(0.0, 1.0);

  // Boundary Conditions
  torch::Tensor d_xy_bc { 
    torch::empty({n_bc * n_data_per_bc, 2}, 
    torch::dtype(torch::kFloat32))
  };

  torch::Tensor u_t_bc { 
    torch::ones({n_bc * n_data_per_bc, 1}, 
    torch::dtype(torch::kFloat32))
  };
  
  for ( int i = 0; i < n_data_per_bc * n_bc; ++i )
  {
    if ( i < n_data_per_bc ) {
      /*x*/ d_xy_bc[i][0] = 0; /*y*/ d_xy_bc[i][1] = rng(rd);
    }

    if ( i < n_data_per_bc * 2 && i >= n_data_per_bc    ) {
      /*x*/ d_xy_bc[i][0] = rng(rd); /*y*/ d_xy_bc[i][1] = 0;
    }

    if ( i < n_data_per_bc * 3 && i >= n_data_per_bc * 2) {
      /*x*/ d_xy_bc[i][0] = 1; /*y*/ d_xy_bc[i][1] = rng(rd);
    }

    if ( i < n_data_per_bc * 4 && i >= n_data_per_bc * 3) {
      /*x*/ d_xy_bc[i][0] = rng(rd); /*y*/ d_xy_bc[i][1] = 1;
    }        
  }

  // Domain
  torch::Tensor d_xy_center {
    torch::ones({n_c, 2},
    torch::dtype(torch::kFloat32))
  };

  for ( int i = 0; i < n_c; ++i ) {
    d_xy_center[i][0] = rng(rd); 
    d_xy_center[i][1] = rng(rd);
  } 

  // std::vector<std::vector<float>> data_x_y_c (n_c, std::vector<float>(2, 0));

  // for ( int i = 0; i < n_c; ++i ) {
  //   data_x_y_c[i][0] = rng(rd); 
  //   data_x_y_c[i][1] = rng(rd);
  // } 

  // torch::Tensor x_y_c = torch::empty({4, 2}).to(device);

  // for ( int i = 0; i < 4; ++i ) {
  //   x_y_c[i][0] = rng(rd); 
  //   x_y_c[i][1] = rng(rd);
  // } 

  // std::cout << x_y_c << std::endl;

  // ------------------------------- PINN Parts ------------------------------- // 
  // auto x = std::vector<float>(2, 0);
  Net model;
  model.to(device);

  // auto tensor1 = torch::tensor({1, 2, 3, 4}, torch::dtype(torch::kInt32));

  // auto x = torch::tensor({1,2}, torch::dtype(torch::kFloat32)).to(device);
  // auto y = model.forward(x_y_c);

  // std::cout << y << std::endl;


  return 0;
}