#include <torch/torch.h>
#include <torch/script.h> // for torch::save

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
  std::mt19937 &, std::uniform_real_distribution<Float> &, 
  size_type, size_type);

void get_bc_dataset_X(
  std::vector<Float> &, 
  std::mt19937 & ,std::uniform_real_distribution<Float> &, 
  size_type, size_type);

torch::Tensor get_pde_loss(torch::Tensor &, torch::Tensor &, torch::Device &);
torch::Tensor get_total_loss(HeatPINN &, torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Device &);


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
  torch::zeros({2 * (NX + NY),  OUT_SIZE}, device)
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
      {NX * NY, IN_SIZE},
      torch::requires_grad()
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
  Integer iter {1}, nsteps {10'000};

  torch::optim::Adam adam_optim(
    net->parameters(), 
    torch::optim::AdamOptions(1E-3)
  );


  while (iter <= nsteps)
  { 
    auto closure = [&](){
      loss_sum = get_total_loss(net, X, X_train, Y_train, device);
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
    << "loss=" << std::setprecision(7) << loss_sum.item<float>()  << "\n"
    << "loss.device().type()=" << loss_sum.device().type() 
    << std::endl;

  torch::save(net, "model.pt");

  torch::Tensor sample { 
    torch::zeros({1, IN_SIZE}, device)
  };

  auto out = net->forward(sample);

  std::cout << out << std::endl;

  return 0;

}














/// @brief 
/// @param obj 
/// @param rde 
/// @param rng 
/// @param nx 
/// @param ny 
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

/// @brief 
/// @param obj 
/// @param rde 
/// @param rng 
/// @param nx 
/// @param ny 
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

/// @brief 
/// @param insize 
/// @param outsize 
/// @param hidsize 
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


/// @brief 
/// @param x 
/// @return 
inline 
torch::Tensor 
HeatPINNImpl::forward(torch::Tensor x)
{
  x = torch::tanh(input(x));
  x = torch::tanh(h0(x));
  x = output(x);

  return x;
}









///
///
/// ====================================== Definitions about PDEs ====================================== ///
///
///




/// @brief 
/// @param u 
/// @param X 
/// @param device 
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

  return torch::mse_loss( du_dxx + du_dyy, f_X );
}

/// @brief 
/// @param net 
/// @param X 
/// @param X_train 
/// @param Y_train 
/// @param device 
torch::Tensor get_total_loss(HeatPINN & net, 
  torch::Tensor & X, torch::Tensor & X_train, torch::Tensor & Y_train, 
  torch::Device & device)
{
  torch::Tensor u { net->forward(X) };

  return torch::mse_loss(net->forward(X_train), Y_train) + get_pde_loss(u, X, device);
}