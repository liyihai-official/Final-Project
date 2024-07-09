#include <torch/torch.h>
#include <iostream>

void basic_auto_operations_example(torch::Device &device)
{
  std::cout << " ================== Running: \"Basic autograd operations\" ================== " << std::endl;

  // create a tensor and set ``torch::requires_grad()`` to track computation with it
  auto x{torch::ones(
      {2, 2},
      torch::requires_grad())};
  std::cout << x << std::endl;

  // Do some tensor operation
  auto y{(x + 2)};
  std::cout << y << std::endl;

  // ``y`` was created as a result of an operation
  std::cout << y.grad_fn()->name() << std::endl;

  // More tensor operation on ``y``
  auto z = (y * y * 3);
  auto out = z.mean();

  std::cout << z << std::endl;
  std::cout << z.grad_fn()->name() << std::endl;
  std::cout << out << std::endl;
  std::cout << out.grad_fn()->name() << std::endl;

  // //
  // std::cout << " \t\t--------- Running: \"backprop operations\" --------- " << std::endl;
  auto a{torch::randn({2, 2})};
  std::cout << a << std::endl;
  a = ((a * 3) * (a - 1));

  std::cout << a.requires_grad() << std::endl;
  a.requires_grad_(true);
  std::cout << a.requires_grad() << std::endl;

  auto b = (a * a).sum();

  std::cout << b.grad_fn()->name() << std::endl;

  // // backprop Print gradients d(out)/dx
  out.backward();

  std::cout << x.grad() << std::endl;

  // Now, an example of Vector-Jacobian product
  x = torch::randn(3, torch::requires_grad()).to(device);

  y = (x * 2).to(device);
  while (y.norm().item<double>() < 2)
  {
    y = y * 2;
  }

  std::cout << y << std::endl;
  std::cout << y.grad_fn()->name() << std::endl;
}

int main()
{
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available())
  {
    device = torch::Device(torch::kCUDA);
    std::cout << "CUDA is avail" << std::endl;
  }
  else
  {
    std::cout << "CUDA is NOT avail" << std::endl;
  }

  basic_auto_operations_example(device);

  return 0;
}