#include <torch/torch.h>
#include <iostream>



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

    return 0;
}