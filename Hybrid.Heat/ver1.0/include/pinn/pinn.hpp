///
/// @file Provides structures of Physics Informed Neural Network
/// 
/// @version 6.0
/// @author LI Yihai
/// 


#ifndef FINAL_PROJECT_PINN_HPP
#define FINAL_PROJECT_PINN_HPP

#pragma once
#include <torch/torch.h>
#include <torch/script.h>


/////

#include <types.hpp>
#include <pinn/types.hpp>
#include <multiarray.hpp>

/////

namespace final_project { namespace PINN {


struct HeatPINNImpl
  : torch::nn::Module
{
  HeatPINNImpl() = default;
  HeatPINNImpl(Integer, Integer, Integer);

  torch::Tensor forward(torch::Tensor &);

  torch::nn::Linear input, h0, h1, h2, output;
  // torch::nn::Conv1d conv0, conv1,
  // torch::nn::Dropout drop0;
}; // struct HeatPINNImpl

TORCH_MODULE(HeatPINN);

torch::Tensor get_pde_loss(torch::Tensor &, torch::Tensor &, torch::Device &);
torch::Tensor get_total_loss(HeatPINN &, torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Device &);



} // namespace PINN
} // namespace final_project




///
///
/// --------------------------- Inline Function Definitions  ---------------------------  ///
///
///

        // conv1 = register_module("conv1", torch::nn::Conv1d(1, 16, /*kernel_size=*/3)); // 1 input channel, 16 output channels, kernel size 3
        // conv2 = register_module("conv2", torch::nn::Conv1d(16, 32, /*kernel_size=*/3)); // 16 input channels, 32 output channels, kernel size 3
        // dropout1 = register_module("dropout1", torch::nn::Dropout(0.5)); // Dropout with 50% probability
        

namespace final_project { namespace PINN {

/// @brief 
/// @param insize 
/// @param outsize 
/// @param hidsize 
inline 
  HeatPINNImpl::HeatPINNImpl(Integer insize, Integer outsize, Integer hidsize)
: 
  input(torch::nn::Linear(insize, hidsize)),
  h0(torch::nn::Linear(hidsize, hidsize)),
  h1(torch::nn::Linear(hidsize, hidsize)),
  // drop0(torch::nn::Dropout(0.5)),
  h2(torch::nn::Linear(hidsize, hidsize)),
  output(torch::nn::Linear(hidsize, outsize))
{
  register_module("Input", input);
  register_module("hidden_0", h0);
  register_module("hidden_1", h1);
  // register_module("drop0", drop0);
  register_module("hidden_2", h2);
  register_module("output", output);
}


/// @brief 
/// @param x 
/// @return 
inline 
  torch::Tensor
  HeatPINNImpl::forward(torch::Tensor & x)
{
  torch::Tensor out = torch::tanh(input(x));
  out = torch::tanh(h0(out));
  out = torch::tanh(h1(out));
  // out = drop0(out);
  out = torch::tanh(h2(out));
  out = output(out);

  return out;
}






/// @brief 
/// @param u 
/// @param X 
/// @param device 
inline 
torch::Tensor get_pde_loss(torch::Tensor & u, torch::Tensor & X, torch::Device & device)
{
  torch::Tensor du_dX {
    torch::autograd::grad(
      {u},
      {X},
      {torch::ones_like(u)},
      true,
      true, 
      true
    )[0]
  };
  torch::Tensor du_dx { du_dX.index({"...", 0}) };
  torch::Tensor du_dy { du_dX.index({"...", 1}) };
  
  torch::Tensor du_dxx {
    torch::autograd::grad(
      {du_dx}, {X}, {torch::ones_like(du_dx),}, 
      true, true, true)[0].index({"...", 0})
  };

  torch::Tensor du_dyy {
    torch::autograd::grad(
      {du_dy}, {X}, {torch::ones_like(du_dy),}, 
      true, true, true)[0].index({"...", 1})
  };

  if (X.size(1) == IN_SIZE_3D)
  {
    torch::Tensor du_dz { du_dX.index({"...", 2}) };
    torch::Tensor du_dzz {
      torch::autograd::grad(
        {du_dz}, {X}, {torch::ones_like(du_dz),},
        true, true, true)[0].index({"...", 2})
    };
    return torch::mean(torch::square(du_dxx + du_dyy + du_dzz));
  } else
  {
    return torch::mean(torch::square(du_dxx + du_dyy));
  }
  // torch::Tensor f_X { torch::zeros_like(du_dxx) };
  // return torch::mse_loss( du_dxx + du_dyy, f_X );
}



/// @brief 
/// @param net 
/// @param X 
/// @param X_train 
/// @param Y_train 
/// @param device 
inline 
torch::Tensor get_total_loss(HeatPINN & net, 
  torch::Tensor & X, torch::Tensor & X_train, torch::Tensor & Y_train, 
  torch::Device & device)
{

  torch::Tensor u { net->forward(X) };
  return torch::mse_loss(net->forward(X_train), Y_train) + get_pde_loss(u, X, device);
}

} // namespace PINN
} // namespace final_project


#endif // end define FINAL_PROJECT_PINN_HPP