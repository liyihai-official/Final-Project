#include <torch/torch.h>
#include <iostream>
#include <chrono>

#include <vector>

struct CustomLoss : torch::nn::Module {
  torch::Tensor forward(  const torch::Tensor & output, 
                          const torch::Tensor & target)
  {return torch::mse_loss(output, target); }

}; // end strct Custom loss : torch::nn::Module


/// @brief This is the module of the customized neural network
struct Net : torch::nn::Module
{

  torch::nn::Linear fc1, fc2, fc3;

  Net()
    : fc1(3 , 10),
      fc2(10, 10),
      fc3(10, 1)
    {
      register_module("fc1", fc1);
      register_module("fc2", fc2);
      register_module("fc3", fc3);
    }


  torch::Tensor forward(torch::Tensor x)
  {
    x = torch::relu(fc1->forward(x));
    x = torch::tanh(fc2->forward(x));

    x = fc3->forward(x);

    return x;
  }

}; // end struct Net

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset>
{

  private:
  std::vector<torch::Tensor> data_;
  std::vector<torch::Tensor> targets_;

  public:
  CustomDataset( const std::vector<std::vector<float>>& data, const std::vector<float>& targets)
  {
    for (const auto& sample: data)
    {
      data_.push_back(torch::tensor(sample).view({3}));
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



template <typename DataLoader>
void train(
  size_t epoch,
  Net& model,
  torch::Device device,
  DataLoader& data_loader,
  torch::optim::Optimizer& optimizer,
  size_t dataset_size
)
{
  model.train();

  size_t batch_idx {0};

  for (auto & batch : data_loader)
  {
    auto data {batch.data.to(device).view({-1,3})}, targets {batch.target.to(device).view({-1,1})};

    data.requires_grad_(true);

    optimizer.zero_grad();

    auto output { model.forward(data) };

    // Compute gradients
    auto grad_output  = torch::ones_like(output);

    auto gradients     = torch::autograd::grad({output}, {data}, 
                                                /* grad_outputs= */ {grad_output}, 
                                                /* Create_graph= */ true,
                                                /* allow_unused= */ true)[0]
                                                .requires_grad_(true);

    auto grad_output_second   = torch::ones_like(gradients);

    auto gradient_second      = torch::autograd::grad({gradients}, {data}, 
                                                      /* grad_outputs= */ {grad_output_second}, 
                                                      /* Create_graph= */ true,
                                                      /* allow_unused= */ true)[0];
    // end of compute gradients

    auto loss { torch::mse_loss(output, targets) + gradient_second.sum() };

    loss.backward();

    optimizer.step();

AT_ASSERT(!std::isnan(loss.template item<float>()));


  if (batch_idx++ % 2 == 0)
    std::printf(
      "\nTrain Epoch : %ld Loss: %.4f",
      epoch, loss.template item<float>()
    );
  }
}



int main() {
  
  torch::DeviceType device_type {torch::cuda::is_available() ? torch::kCUDA : torch::kCPU};
  std::cout << "Running on: " << device_type << std::endl;

  torch::Device device(device_type);

  Net model;
  model.to(device);

  std::vector<std::vector<float>> data = 
    {
      {1.0, 2.0, 3.0},
      {4.0, 5.0, 0.0},
      {3.0, 1.0, 1.0},
      {2.0, 1.0, 1.0}
    };
  
  std::vector<float> targets = {0.1, 0.2, 0.3, 0.1};


  auto train_dataset = CustomDataset(data, targets)
    .map(torch::data::transforms::Normalize<>(0.0, 1.0))
    .map(torch::data::transforms::Stack<>());

  const size_t train_dataset_size = train_dataset.size().value();

  auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
    std::move(train_dataset),
    2
  );

  std::cout << "Train Dataset Size: " << train_dataset_size << std::endl;

  torch::optim::SGD optimizer(
    model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5)
  );

  // for (auto& batch : *data_loader)
  // {
  //   auto inputs = batch.data;
  //   auto targets = batch.target;

  //   std::cout << "Inputs: "   << inputs   << std::endl;
  //   std::cout << "Targets: "  << targets  << std::endl;
  //   std::cout << std::endl;
  // }

  for (size_t epoch = 1; epoch <= 200; ++epoch)
  {
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
  }
  
  // std::cout << " \n \t\t THIS is THE GAP \n " << std::endl;

  // torch::Device device(torch::kCPU);
  // if (torch::cuda::is_available()) 
  //   { device = torch::Device(torch::kCUDA); }
  
  // Net net;
  // net.to(device);

  // CustomLoss custom_loss;

  // torch::optim::SGD optimizer(net.parameters(), torch::optim::SGDOptions(0.01));

  // auto inputs =   torch::rand({10,10}).to(device);
  // auto targets =  torch::randn({10,1}).to(device);
  
  // auto t1 {std::chrono::steady_clock::now()};
  // for (std::size_t epoch = 0; epoch < 10; ++epoch)
  // {
  //   auto outputs = net.forward(inputs);
  //   auto loss = custom_loss.forward(outputs, targets);


  //   std::cout << "Epoch [" << epoch << "]" << " Loss " << loss.item<float>() << std::endl;


  //   optimizer.zero_grad();
  //   loss.backward();
  //   optimizer.step();
  // }
  // auto t2 {std::chrono::steady_clock::now()};
  // std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << std::endl;


  return 0;
}