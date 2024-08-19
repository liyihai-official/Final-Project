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

  torch::Tensor forward(torch::Tensor);

  torch::nn::Linear input, h0, h1, h2, output;
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
  h2(torch::nn::Linear(hidsize, hidsize)),
  output(torch::nn::Linear(hidsize, outsize))
{
  register_module("Input", input);
  register_module("hidden_0", h0);
  register_module("hidden_1", h1);
  register_module("hidden_2", h2);
  register_module("output", output);
}


/// @brief 
/// @param x 
/// @return 
inline 
  torch::Tensor 
  HeatPINNImpl::forward(torch::Tensor x)
{
  x = torch::tanh(input(x));
  x = torch::tanh(h0(x));
  x = torch::tanh(h1(x));
  x = torch::tanh(h2(x));
  x = output(x);

  return x;
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

  torch::Tensor f_X { torch::zeros_like(du_dxx) };

  // return torch::mse_loss( du_dxx + du_dyy, f_X );
  return torch::mean(torch::square(du_dxx + du_dyy));
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