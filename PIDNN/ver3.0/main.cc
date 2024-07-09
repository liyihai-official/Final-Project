#include <torch/torch.h>
#include <iostream>
#include <chrono>

struct CustomLoss : torch::nn::Module {
    torch::Tensor forward(  const torch::Tensor & output, 
                            const torch::Tensor & target)



    {return torch::mse_loss(output, target); }

}; // end strct Custom loss : torch::nn::Module


struct Net : torch::nn::Module {
    Net() 
    {
        fc1 = register_module("fc1", torch::nn::Linear(10, 500));
        fc2 = register_module("fc2", torch::nn::Linear(500, 500));
        fc3 = register_module("fc3", torch::nn::Linear(500, 1));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);

        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

}; // end struct Net : torch::nn::Module

int main() {
    // Check if CUDA is available
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Using GPU." << std::endl;

        // Create a tensor and move it to the GPU
        torch::Tensor tensor = torch::rand({2, 3}).cuda();
        std::cout << "Tensor on GPU: " << tensor << std::endl;

        // Perform a simple operation
        torch::Tensor result = tensor + tensor;
        std::cout << "Result of tensor addition on GPU: " << result << std::endl;
    } else {
        std::cout << "CUDA is not available. Using CPU." << std::endl;

        // Create a tensor on the CPU
        torch::Tensor tensor = torch::rand({2, 3});
        std::cout << "Tensor on CPU: " << tensor << std::endl;

        // Perform a simple operation
        torch::Tensor result = tensor + tensor;
        std::cout << "Result of tensor addition on CPU: " << result << std::endl;
    }


    std::cout << " \n \t\t THIS is THE GAP \n " << std::endl;

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) 
        { device = torch::Device(torch::kCUDA); }
    
    Net net;
    net.to(device);

    CustomLoss custom_loss;

    torch::optim::SGD optimizer(net.parameters(), torch::optim::SGDOptions(0.01));

    auto inputs =   torch::rand({10,10}).to(device);
    auto targets =  torch::randn({10,1}).to(device);
    
    auto t1 {std::chrono::steady_clock::now()};
    for (std::size_t epoch = 0; epoch < 10000000; ++epoch)
    {
        auto outputs = net.forward(inputs);
        auto loss = custom_loss.forward(outputs, targets);


        std::cout << "Epoch [" << epoch << "]" << " Loss " << loss.item<float>() << std::endl;


        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
    auto t2 {std::chrono::steady_clock::now()};
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << std::endl;
    


    return 0;
}