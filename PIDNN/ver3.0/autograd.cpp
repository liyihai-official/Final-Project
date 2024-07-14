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
  std::cout << " ================== Running: \"Basic Vector-Jacobian product \" ================== " << std::endl;

  x = torch::randn(3, torch::requires_grad());

  y = (x * 2);
  while (y.norm().item<double>() < 2)
  {
    y = y * 2;
  }

  std::cout << y << std::endl;
  std::cout << y.grad_fn()->name() << std::endl;

  auto v {torch::tensor({0.1, 1.0, 0.0001}, torch::kFloat)};
  y.backward(v);

  std::cout << x.grad() << std::endl;

  // Also can 'STOP' autograd tracking history on tensors that require gradients either by putting `torch::NoGradGuard` in a  code block
  std::cout << x.requires_grad() << std::endl;
  std::cout << x.pow(2).requires_grad() << std::endl;

  {
    torch::NoGradGuard no_grad;
    std::cout << x.pow(2).requires_grad() << std::endl;
  }

  std::cout << x.requires_grad() << std::endl;
  y = x.detach();
  std::cout << y.requires_grad() << std::endl;
  std::cout << x.eq(y).all().item<bool>() << std::endl;
}



void compute_higher_order_gradients_example(torch::Device & device)
{
  std::cout << " ================== Running: \"Computing Higher-Order gradients in C++ (torch) \" ================== " << std::endl;

  auto model {torch::nn::Linear(4, 3)};

  auto input {torch::randn({3,4}).requires_grad_(true)};
  auto output {model(input)};

  auto target {torch::randn({3,3})};

  auto loss = torch::nn::MSELoss()(output, target);

  // Use norm of gradients as penalty
  auto grad_output  = torch::ones_like(output);
  auto gradient     = torch::autograd::grad({output}, {input}, 
                                            /* grad_outputs=*/{grad_output},
                                            /* create_graph=*/true)[0];

  auto gradient_penalty = torch::pow(
    (gradient.norm(2, /*dims=*/1)-1), 
    2
  ).mean();

  auto combined_loss = loss + gradient_penalty;

  combined_loss.backward();

  std::cout << input.grad() << std::endl;

}


/////////

class my_Linear_Function : public torch::autograd::Function<my_Linear_Function>
{
  public:
  
  static torch::Tensor forward(
    torch::autograd::AutogradContext *ctx, 
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias = torch::Tensor()
  ) 
  {
    ctx->save_for_backward({input, weight, bias});

    auto output = input.mm(weight.t());

    if (bias.defined())
    {
      output += bias.unsqueeze(0).expand_as(output);
    }

    return output;
  }



  static torch::autograd::tensor_list backward(  
    torch::autograd::AutogradContext *ctx,
    torch::autograd::tensor_list grad_outputs)
  {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto bias = saved[2];

    auto grad_output = grad_outputs[0];
    auto grad_input = grad_output.mm(weight);
    auto grad_weight = grad_output.t().mm(input);

    auto grad_bias = torch::Tensor();
    if (bias.defined()) {
      grad_bias = grad_output.sum(0);
    }

    return {grad_input, grad_weight, grad_bias};
  }

};

class MulConstant : public torch::autograd::Function<MulConstant> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor tensor, double constant) {
    // ctx is a context object that can be used to stash information
    // for backward computation
    ctx->saved_data["constant"] = constant;
    return tensor * constant;
  }

  static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::tensor_list grad_outputs) {
    // We return as many input gradients as there were arguments.
    // Gradients of non-tensor arguments to forward must be `torch::Tensor()`.
    return {grad_outputs[0] * ctx->saved_data["constant"].toDouble(), torch::Tensor()};
  }
};

void custom_autograd_function_example(torch::Device & device)
{
  std::cout << " ================== Running: \"Using Custom Autograd Function in C++ (torch)\" ================== " << std::endl;

  {
    auto x      = torch::randn({2,3}).requires_grad_();
    auto weight = torch::randn({4,3}).requires_grad_();
    
    auto y      = my_Linear_Function::apply(x, weight);

    y.sum().backward();


    std::cout << x.grad() << std::endl;
    std::cout << weight.grad() << std::endl;
  }

  {
    auto x = torch::randn({2}).requires_grad_();
    auto y = MulConstant::apply(x, 5.5);

    y.sum().backward();
    std::cout << x.grad() << std::endl;
  }
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

  // basic_auto_operations_example(device);
  // compute_higher_order_gradients_example(device);
  custom_autograd_function_example(device);




  return 0;
}