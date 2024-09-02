#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

#include <Eigen/Dense>

// --------------------------------------------------------------------------------------------------------------
// Activation Functions

double relu(double);
double relu_derivative(double);

double mytanh(double);
double mytanh_derivative(double);

double sigmoid(double);
double sigmoid_derivate(double);



double relu(double x)            { return std::max(0.0, x); }
double relu_derivative(double x) { return x > 0 ? 1.0 : 0.0; }

double mytanh(double x)             { return std::tanh(x); }
double mytanh_derivative(double x)  { return 1 - std::tanh(x) * std::tanh(x); }

double sigmoid(double x)              { return 1.0 / (1.0 + std::exp(-x)); }           
double sigmoid_derivate(double x)     { return x * ( 1 - x ); }

// --------------------------------------------------------------------------------------------------------------

/// @brief The object of Dense Layer
class DenseLayer
{
  private:
  Eigen::MatrixXd W;
  Eigen::VectorXd b, z, a;

  public:
  DenseLayer() = default;
  DenseLayer(int, int);

  Eigen::VectorXd feedforward(const Eigen::VectorXd & , std::string activate_function);
  Eigen::VectorXd backward_propagation( const Eigen::VectorXd &, std::string activate_function);
  void update_trainable_weights(const Eigen::VectorXd & delta, const Eigen::VectorXd & a_prev, const double learning_rate);

  public:
  // Gets 
  Eigen::MatrixXd get_Weight() { return W; }
  Eigen::VectorXd get_Bias()   { return b; }

  Eigen::MatrixXd & get_ref_Weight() { return W; }
  Eigen::VectorXd & get_ref_Bias()   { return b; }

  Eigen::VectorXd & get_ref_a() { return a; }
  Eigen::VectorXd & get_ref_z() { return z; }

  Eigen::MatrixXd get_derivative(std::string activate_function) { return     z.unaryExpr(
      activate_function == "relu" ? [](double x){ return mytanh_derivative(x); } : 
                                    [](double x) { return 1.0; } // Here is the Temporary Activation Function
    ).asDiagonal() * W; }
};

/// @brief Initialize the Weights and Bias matrices and vectors By Xavier Strategy
/// @param inSize Input shape (1, inSize)
/// @param outSize Output shape (1, outSize)
/// @details The Weight Matrix has shape (outSize, inSize), and the Bias has the same 
///         shape with Output shape (1, outSize).
DenseLayer::DenseLayer(int inSize, int outSize)
{
  // Inits of Weights and Bias
  W = Eigen::MatrixXd::Random(outSize, inSize) * std::sqrt(1.0 / inSize);
  b = Eigen::VectorXd::Random(outSize);

}


/// @brief Forward Propagation of A single Layer (Dense Layer)
/// @param input The input prompted from Previous Layer 
/// @return The output activated Data.
Eigen::VectorXd DenseLayer::feedforward( const Eigen::VectorXd & input, std::string activate_function)
{
  z = W * input + b;
  a = z.unaryExpr( 
    activate_function == "relu" ? [](double x) { return mytanh(x); } : 
                                  [](double x) { return x; } // Here is the Temporary Activation Function
  );
  return a;
}

/// @brief BackWard Propagation of A single Layer (Dense Layer)
/// @param Delta The difference Prompted from Previous (backward) layer.
/// @return Return the Delta_Out, of this Layer.
Eigen::VectorXd DenseLayer::backward_propagation( const Eigen::VectorXd & Delta, std::string activate_function)
{
  Eigen::VectorXd Delta_Out = Delta.cwiseProduct(
    z.unaryExpr(
      activate_function == "relu" ? [](double x){ return mytanh_derivative(x); } : 
                                    [](double x) { return 1.0; } // Here is the Temporary Activation Function
    )
  );
  return Delta_Out;
}

void DenseLayer::update_trainable_weights(const Eigen::VectorXd & delta, const Eigen::VectorXd & a_prev, const double learning_rate)
{
  std::cout << "Hello FUCK \n";
  W += learning_rate * delta * a_prev.transpose();
  b += learning_rate * delta;
}

// --------------------------------------------------------------------------------------------------------------

/// @brief The Body of Neural Network
class NeuralNetwork
{
  private:
  Eigen::VectorXd a1, a2;
  
  DenseLayer HiddenLayer_1;
  DenseLayer OutputLayer;

  public:
  NeuralNetwork(int, int, int);

  Eigen::VectorXd feedforward(const Eigen::VectorXd &);

  void train( const std::vector<Eigen::VectorXd>& , 
              const std::vector<Eigen::VectorXd>& , 
              double, int);

};


/// @brief Initialize the Weights and Bias matrices and vectors By Xavier Strategy
/// @param inSize Input shape (1, inSize)
/// @param hidSize Hidden shape (1, hidSize)
/// @param outSize Output shape (1, outSize)
NeuralNetwork::NeuralNetwork(int inSize, int hidSize, int outSize)
{
  // Inits of Weights and Bias With Layers Objects
  HiddenLayer_1 = DenseLayer(inSize, hidSize);
  OutputLayer   = DenseLayer(hidSize, outSize);

}

/// @brief Forward propagation of the Network with sample data (labeled) as input.
/// @param input_data The input sample data.
Eigen::VectorXd NeuralNetwork::feedforward( const Eigen::VectorXd & input_data )
{

  // With Layer Objects 
  a1 = HiddenLayer_1.feedforward(input_data, "relu");
  a2 =   OutputLayer.feedforward(a1, "none");
  
  return a2;
}

/// @brief 
/// @param inputS_Boundary 
/// @param labelS_Boundary 
/// @param learning_rate 
/// @param epochS 
void NeuralNetwork::train(const std::vector<Eigen::VectorXd>& inputS_Boundary, 
                          const std::vector<Eigen::VectorXd>& labelS_Boundary, 
                          // const std::vector<Eigen::VectorXd>& inputS_Central,  
                          double learning_rate, int epochS)
{
  for (int epoch = 0; epoch < epochS; ++epoch)
  {
    double total_Err {0.0};

    // 累积梯度
    Eigen::MatrixXd total_Delta2 = Eigen::MatrixXd::Zero(OutputLayer.get_ref_Weight().rows(), OutputLayer.get_ref_Weight().cols());
    Eigen::MatrixXd total_Delta1 = Eigen::MatrixXd::Zero(HiddenLayer_1.get_ref_Weight().rows(), HiddenLayer_1.get_ref_Weight().cols());

    Eigen::MatrixXd total_Delta2_Bias = Eigen::VectorXd::Zero(OutputLayer.get_ref_Weight().rows());
    Eigen::MatrixXd total_Delta1_Bias = Eigen::VectorXd::Zero(HiddenLayer_1.get_ref_Weight().rows());

    for (std::size_t i = 0; i < inputS_Boundary.size(); ++i) // Feed the Network Sample by Sample
    {
      // Front Propagation
      Eigen::VectorXd input  {inputS_Boundary[i]}, label {labelS_Boundary[i]};
      Eigen::VectorXd output {feedforward(input)};

      // Calculate Error
      Eigen::VectorXd Err = output - label;
      total_Err += Err.squaredNorm(); //                        Temporary Loss Function, L2 Norm.

      Eigen::MatrixXd Delta_NN = HiddenLayer_1.get_derivative("relu");
      Eigen::MatrixXd Hessian = 0.5 * (Delta_NN.transpose() * OutputLayer.get_ref_Weight().transpose() * OutputLayer.get_ref_Weight() * Delta_NN);

      total_Err += Hessian.trace() * Hessian.trace();
      
      // Backward Propagation
      Eigen::VectorXd Delta2 = OutputLayer.backward_propagation(2.0 * Err / inputS_Boundary.size(), "none");
      Eigen::VectorXd Delta1 = HiddenLayer_1.backward_propagation(OutputLayer.get_ref_Weight().transpose() * Delta2, "relu");

      // 累积梯度
      total_Delta2 += Delta2 * HiddenLayer_1.get_ref_a().transpose();
      total_Delta2_Bias += Delta2;

      total_Delta1 += Delta1 * input.transpose();
      total_Delta1_Bias += Delta1;

    }
    
    // // 在处理完所有样本后更新权重和偏置
    // OutputLayer.get_ref_Weight() -= total_Delta2 / inputS_Boundary.size() * learning_rate;
    // OutputLayer.get_ref_Bias()   -= total_Delta2_Bias / inputS_Boundary.size() * learning_rate;

    // HiddenLayer_1.get_ref_Weight() -= total_Delta1 / inputS_Boundary.size() * learning_rate;
    // HiddenLayer_1.get_ref_Bias()   -= total_Delta1_Bias / inputS_Boundary.size() * learning_rate;

    // 更新权重和偏置
    Eigen::MatrixXd Hessian_W2 = Eigen::MatrixXd::Zero(OutputLayer.get_ref_Weight().rows(), OutputLayer.get_ref_Weight().cols());
    Eigen::MatrixXd Hessian_W1 = Eigen::MatrixXd::Zero(HiddenLayer_1.get_ref_Weight().rows(), HiddenLayer_1.get_ref_Weight().cols());
   
    // 计算Hessian矩阵对权重的影响（假设为identity matrix乘以某个因子）
    Hessian_W2.setIdentity();
    Hessian_W1.setIdentity();
    
    double hessian_factor = 1e-4; // 可调节因子，控制Hessian部分的更新幅度
    
    // std::cout << total_Delta2_Bias << "\n----------------------------------------------------- \n" << std::endl;
    OutputLayer.get_ref_Weight() -= (total_Delta2 / inputS_Boundary.size() + hessian_factor * Hessian_W2) * learning_rate;
    OutputLayer.get_ref_Bias()   -= (total_Delta2_Bias / inputS_Boundary.size() + hessian_factor * Hessian_W2.colwise().sum().rowwise().sum()) * learning_rate;

    HiddenLayer_1.get_ref_Weight() -= (total_Delta1 / inputS_Boundary.size() + hessian_factor * Hessian_W1) * learning_rate;
    HiddenLayer_1.get_ref_Bias()   -= (total_Delta1_Bias / inputS_Boundary.size() + hessian_factor * Hessian_W1.rowwise().sum()) * learning_rate;


    // Verbose for showing details during Training.
    if (epoch % 1000 == 0) std::cout << "Epoch " << epoch << " Total Error: " << total_Err << std::endl;
  }
}


int main ()
{

  int inputS_Boundaryhape {2}, hiddenShape {5}, outputShape {1};

  std::cout 
  << inputS_Boundaryhape  << " "
  << hiddenShape << " "
  << outputShape << " "
  << std::endl;

  NeuralNetwork nn(inputS_Boundaryhape, hiddenShape, outputShape);
  // std::cout << "Before Train: \n" << nn.get_Weight1() << std::endl;

  std::vector<Eigen::VectorXd> inputS_Boundary;
  //  = {
  //   Eigen::VectorXd::Zero(2),
  //   Eigen::VectorXd::Ones(2)
  // };


  std::vector<Eigen::VectorXd> labelS_Boundary;
  //  = {
  //   Eigen::VectorXd::Ones(1),
  //   Eigen::VectorXd::Zero(1)
  // };

  for (int i = 0; i < 100; ++i) {
    // Generate a random 2D vector
    Eigen::VectorXd temp = Eigen::VectorXd::Random(2);
    inputS_Boundary.push_back(temp);

    // Generate a label based on the first element of the vector
    if (temp(0) * temp(0) + temp(1) * temp(1) <= 0.3) 
    {
      labelS_Boundary.push_back(Eigen::VectorXd::Ones(1)); // Label 1
    } else {
      labelS_Boundary.push_back(Eigen::VectorXd::Zero(1)); // Label 0
    }
  }

  nn.train(inputS_Boundary, labelS_Boundary, 0.001, 20000);

  std::cout << " ---------------------------------------------------- \n";
  std::cout << nn.feedforward(Eigen::VectorXd::Zero(2)) << std::endl; // A test for Forward Propagation
  std::cout << nn.feedforward(Eigen::VectorXd::Ones(2)) << std::endl; // A test for Forward Propagation

  // std::cout << "After Train: \n" << nn.get_Weight1() << std::endl;

  std::ofstream op("result.txt");

  std::vector<Eigen::VectorXd> Test, labelS_Test;
  for (int i = 0; i < 500; ++i) {
    // 生成一个二维随机向量
    Eigen::VectorXd temp = Eigen::VectorXd::Random(2);

    // 将该随机向量添加到 Test 向量中
    Test.push_back(temp);

    // 根据向量的第一个元素决定标签
    if (temp(0) * temp(0) + temp(1) * temp(1) <= 0.3) {
      // 若第一个元素小于或等于 0，标签为 1
      labelS_Test.push_back(Eigen::VectorXd::Ones(1));
    } else {
      // 若第一个元素大于 0，标签为 0
      labelS_Test.push_back(Eigen::VectorXd::Zero(1));
    }
  }

  // 打印生成的数据和标签
  // std::cout << "\nGenerated Data and Labels:" << std::endl;
  // for (int i = 0; i < 50; ++i) {
  //   std::cout << "Test[" << i << "]: " << Test[i].transpose()
  //             << "\tlabelS_Test[" << i << "]: " << labelS_Test[i] << std::endl;
  // }

  for (int i = 0; i < 500; ++i)
  {
    for (int k = 0; k < Test[i].size(); ++k)
    {
      op << Test[i][k];
      op << " ";
    }

    op << labelS_Test[i] << " ";
    op << nn.feedforward(Test[i]);
    op << "\n";
  }

  return 0;
}





