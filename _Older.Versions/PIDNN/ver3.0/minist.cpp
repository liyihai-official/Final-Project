#include <torch/torch.h>

#include <cstdio>
#include <cstddef>

#include <iostream>
#include <string>
#include <vector>


#include <chrono>

const char * kDataRoot = "../py";
const int64_t kTrainBatchSize = 64;
const int64_t kTestBatchSize  = 1000;
const int64_t kNumberofEpochs = 10;
const int64_t kLogInterval    = 10;

struct Net : torch::nn::Module
{

  torch::nn::Conv2d conv1; 
  torch::nn::Conv2d conv2;

  torch::nn::Dropout2d conv2_drop;

  torch::nn::Linear fc1;
  torch::nn::Linear fc2;


  Net()
    : conv1(torch::nn::Conv2dOptions(1 , 100, 5)),
      conv2(torch::nn::Conv2dOptions(100, 20, 5)),
      fc1(320, 50),
      fc2(50, 10)
    {
      register_module("Conv1", conv1);
      register_module("Conv2", conv2);
      register_module("Conv2_drop", conv2_drop);
      register_module("fc1", fc1);
      register_module("fc2", fc2);
    }

  torch::Tensor forward (torch::Tensor x)
  {
    x = torch::relu(
      torch::max_pool2d(conv1->forward(x), 2)
    );

    x = torch::relu(
      torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2)
    );

    x = x.view( {-1, 320} );

    x = torch::relu(fc1->forward(x));

    x = torch::dropout(x, 0.5, is_training());

    x = fc2->forward(x);

    return torch::log_softmax(x, 1);

  }

};

template <typename Dataloader>
void train(size_t, Net&, torch::Device, Dataloader&, torch::optim::Optimizer&, size_t);

template <typename Dataloader>
void test(Net&, torch::Device, Dataloader&, size_t);


int main(int argc, char ** argv)
{
  torch::manual_seed(1);
  

  torch::DeviceType device_type;
  if (torch::cuda::is_available())
  {
    std::cout << "CUDA available! Training on GPU. " << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);
  
  ////////////

  Net model;
  model.to(device);


  auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
      .map(torch::data::transforms::Stack<>());

  auto test_dataset = torch::data::datasets::MNIST(
    kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
      .map(torch::data::transforms::Stack<>());

  const size_t train_dataset_size = train_dataset.size().value(), test_dataset_size = test_dataset.size().value();

  std::cout << "Train dataset size: \t"   << train_dataset_size 
            << "\nTest dataset size: \t"  << test_dataset_size << std::endl;

  auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
    std::move(train_dataset), kTrainBatchSize
  );

  auto test_loader = torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

  torch::optim::SGD optimizer(
    model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5)
  );

  auto s0 = std::chrono::steady_clock::now();
  for (size_t epoch = 1; epoch <= kNumberofEpochs; ++epoch)
  {
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
    test(model, device, *test_loader, test_dataset_size);
  }
  auto s1 = std::chrono::steady_clock::now();

  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(s1 - s0).count() << std::endl;

  return 0;

}


template <typename Dataloader>
void train(
  size_t epoch, 
  Net& model, 
  torch::Device device, 
  Dataloader& data_loader, 
  torch::optim::Optimizer& optimizer, 
  size_t dataset_size)
{
  model.train();

  size_t batch_idx {0};

  for (auto& batch : data_loader)
  {
    auto data {batch.data.to(device)}, targets {batch.target.to(device)};
    data.requires_grad_(true);

    optimizer.zero_grad();

    auto output { model.forward(data) };

    auto loss {torch::nll_loss(output, targets)};

    auto grad_output = torch::ones_like(output);
    auto gradients = torch::autograd::grad({output}, {data}, {grad_output}, true);

    auto grad_output_second = torch::ones_like(gradients[0]);
    auto gradient_second    = torch::autograd::grad({gradients[0]}, {data},
                                                    /* grad_outputs=*/{grad_output_second},
                                                    /* Create_graph=*/true);
    
    std::cout << "Gradients : " << gradients[0].to(torch::kCPU) << std::endl;

AT_ASSERT(!std::isnan(loss.template item<float>()));

    loss.backward();
    optimizer.step();

    if (batch_idx++ % kLogInterval == 0)
    {
      std::printf(
        "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
        epoch,
        batch_idx * batch.data.size(0),
        dataset_size,
        loss.template item<float>()
      );
    }


  }
}












template <typename DataLoader>
void test(
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model.forward(data);
    test_loss += torch::nll_loss(
                     output,
                     targets,
                     /*weight=*/{},
                     torch::Reduction::Sum)
                     .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  test_loss /= dataset_size;
  std::printf(
      "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
      test_loss,
      static_cast<double>(correct) / dataset_size);
}