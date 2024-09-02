#include <torch/torch.h>
#include <iostream>
#include <vector>

// 定义PINN模型
struct PINN : torch::nn::Module {
    torch::nn::Linear layer1{nullptr}, layer2{nullptr}, layer3{nullptr};

    PINN() {
        layer1 = register_module("layer1", torch::nn::Linear(2, 20));
        layer2 = register_module("layer2", torch::nn::Linear(20, 20));
        layer3 = register_module("layer3", torch::nn::Linear(20, 1));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(layer1->forward(x));
        x = torch::relu(layer2->forward(x));
        x = layer3->forward(x);
        return x;
    }
};

// 定义热传导方程的物理损失
torch::Tensor physics_loss(torch::nn::Module &model, torch::Tensor x, torch::Tensor t) {
    torch::Tensor xt = torch::cat({x, t}, 1).requires_grad_(true);
    torch::Tensor u = model.forward(xt);
    torch::Tensor u_t = torch::autograd::grad({u.sum()}, {t}, /*grad_outputs=*/{}, /*retain_graph=*/true)[0];
    torch::Tensor u_x = torch::autograd::grad({u.sum()}, {x}, /*grad_outputs=*/{}, /*retain_graph=*/true)[0];
    torch::Tensor u_xx = torch::autograd::grad({u_x.sum()}, {x}, /*grad_outputs=*/{}, /*retain_graph=*/true)[0];
    torch::Tensor f = u_t - 0.01 * u_xx; // 热传导方程：u_t = 0.01 * u_xx
    return torch::mean(torch::pow(f, 2));
}

int main() {
    // 设置设备
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
    }

    // 初始化模型和优化器
    PINN model;
    model.to(device);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));

    // 训练数据
    int num_samples = 100;
    auto x = torch::rand({num_samples, 1}, device);
    auto t = torch::rand({num_samples, 1}, device);

    // 训练模型
    int num_epochs = 1000;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model.train();
        optimizer.zero_grad();

        // 计算损失
        auto loss = physics_loss(model, x, t);
        std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "], Loss: " << loss.item<double>() << std::endl;

        // 反向传播和优化
        loss.backward();
        optimizer.step();
    }

    std::cout << "Training completed." << std::endl;
    return 0;
}
