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

class CustomDataset_x_y
: public torch::data::datasets::Dataset<CustomDataset_x_y>
  {
    private:
    std::vector<torch::Tensor> data_;
    std::vector<torch::Tensor> targets_;

    public:
    CustomDataset_x_y(  const std::vector<std::vector<float>>&  data, 
                        const std::vector<float>&               targets)
    {
      for (const auto& sample: data)
      {
        data_.push_back(torch::tensor(sample).view({2}));
      }

      for (const auto& target : targets)
      {
        targets_.push_back(torch::tensor(target).view({1}));
      }
    }

    torch::data::Example<> get(size_t index) override {
      return {data_[index], targets_[index]};
    }

    torch::optional<size_t> size()
    const override
    { return data_.size(); }
  };

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

  // Boundary 
  std::vector<std::vector<float>> data_x_y_d 
                              (n_bc * n_data_per_bc, std::vector<float>(2, 0));
  std::vector<float>              data_t_d 
                              (n_bc * n_data_per_bc, 1);

  for ( int i = 0; i < n_data_per_bc * n_bc; ++i )
  {
    if ( i < n_data_per_bc ) {
      /*x*/ data_x_y_d[i][0] = 0; /*y*/ data_x_y_d[i][1] = rng(rd);
    }

    if ( i < n_data_per_bc * 2 && i >= n_data_per_bc    ) {
      /*x*/ data_x_y_d[i][0] = rng(rd); /*y*/ data_x_y_d[i][1] = 0;
    }

    if ( i < n_data_per_bc * 3 && i >= n_data_per_bc * 2) {
      /*x*/ data_x_y_d[i][0] = 1; /*y*/ data_x_y_d[i][1] = rng(rd);
    }

    if ( i < n_data_per_bc * 4 && i >= n_data_per_bc * 3) {
      /*x*/ data_x_y_d[i][0] = rng(rd); /*y*/ data_x_y_d[i][1] = 1;
    }        
  }

  // Domain
  std::vector<std::vector<float>> data_x_y_c (n_c, std::vector<float>(2, 0));

  for ( int i = 0; i < n_c; ++i ) {
    data_x_y_c[i][0] = rng(rd); 
    data_x_y_c[i][1] = rng(rd);
  } 

  torch::Tensor x_y_c = torch::empty({4, 2}).to(device);

  for ( int i = 0; i < 4; ++i ) {
    x_y_c[i][0] = rng(rd); 
    x_y_c[i][1] = rng(rd);
  } 

  std::cout << x_y_c << std::endl;

  // ------------------------------- PINN Parts ------------------------------- // 
  // auto x = std::vector<float>(2, 0);
  Net model;
  model.to(device);

  // auto tensor1 = torch::tensor({1, 2, 3, 4}, torch::dtype(torch::kInt32));

  // auto x = torch::tensor({1,2}, torch::dtype(torch::kFloat32)).to(device);
  auto y = model.forward(x_y_c);

  std::cout << y << std::endl;
  return 0;
}