#include <iostream>
#include <vector>
#include <cmath>

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
double mytanh_derivative(double x)  { return 1.0 / ( 1.0 + std::tanh(x) * std::tanh(x)); }

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
  // std::cout << Delta << std::endl;
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
  
  W += learning_rate * delta * a_prev.transpose();
  b += learning_rate * delta;
}

// --------------------------------------------------------------------------------------------------------------

/// @brief The Body of Neural Network
class NeuralNetwork
{
  private:
  Eigen::MatrixXd W1,     W2;
  Eigen::VectorXd b1,     b2;
  Eigen::VectorXd z1, a1, z2, a2;
  
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
/// @param inputS 
/// @param labelS 
/// @param learning_rate 
/// @param epochS 
void NeuralNetwork::train(const std::vector<Eigen::VectorXd>& inputS, 
                          const std::vector<Eigen::VectorXd>& labelS, 
                          double learning_rate, int epochS)
{
  for (int epoch = 0; epoch < epochS; ++epoch)
  {
    double total_Err {0.0};
    for (std::size_t i = 0; i < inputS.size(); ++i) // Feed the Network Sample by Sample
    {
      // Front Propagation
      Eigen::VectorXd input  {inputS[i]}, label {labelS[i]};

      
      Eigen::VectorXd output {feedforward(input)};

  
      
      // Calculate Error
      Eigen::VectorXd Err = label - output;
      total_Err += Err.norm(); //                        Temporary Loss Function, L2 Norm.

      // Backward Propagation
      Eigen::VectorXd Delta2 = OutputLayer.backward_propagation(Err, "none");
      Eigen::VectorXd Delta1 = HiddenLayer_1.backward_propagation(OutputLayer.get_ref_Weight().transpose() * Delta3, "relu");
      
      // Update Weights and Biases
      OutputLayer.update_trainable_weights(   Delta2,   HiddenLayer_1.get_ref_a(),  learning_rate);
      HiddenLayer_1.update_trainable_weights( Delta1,   input,                      learning_rate);

    }

    // Verbose for showing details during Training.
    if (epoch % 100 == 0) std::cout << "Epoch " << epoch << " Total Error: " << total_Err << std::endl;
  }
}


int main ()
{
  

  int inputShape {2}, hiddenShape {5}, outputShape {1};

  std::cout 
  << inputShape  << " "
  << hiddenShape << " "
  << outputShape << " "
  << std::endl;

  NeuralNetwork nn(inputShape, hiddenShape, outputShape);
  // std::cout << "Before Train: \n" << nn.get_Weight1() << std::endl;

  std::vector<Eigen::VectorXd> inputS = {
    Eigen::VectorXd::Zero(2),
    Eigen::VectorXd::Ones(2)
  };

  std::vector<Eigen::VectorXd> labelS = {
    Eigen::VectorXd::Ones(1),
    Eigen::VectorXd::Zero(1)
  };


  nn.train(inputS, labelS, 0.01, 4000);

  std::cout << nn.feedforward(Eigen::VectorXd::Zero(2)) << std::endl; // A test for Forward Propagation
  std::cout << nn.feedforward(Eigen::VectorXd::Ones(2)) << std::endl; // A test for Forward Propagation

  // std::cout << "After Train: \n" << nn.get_Weight1() << std::endl;

  return 0;
}





