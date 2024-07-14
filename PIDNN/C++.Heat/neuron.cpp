#ifndef _NEURON_CPP_
#define _NEURON_CPP_

#pragma once 
#include <memory>
#include <random>
#include <functional>

#include "../../MPI.Heat/C++.Heat/ver3.0/multi_array/base.cpp"

namespace NeuralNetwork
{

class Neuron
{
  public:
  Neuron();
  Neuron(double value);

  void activate();
  void derivative();

  public:
  void setValue(double const & value) { this->value = value; activate(); derivative(); }
  double getVal() {  return this->value;  }

  double get_activated_value() { return this->activateValue; }
  double get_derivative_Value() { return this->derivativeValue; }

  private:
  double value, activateValue, derivativeValue;
}; // class Neuron



// class NeuronMats
// {
//   public:
//   /// @brief Default Constructor
//   NeuronMats();

//   /// @brief Constructor
//   /// @param Rows Row of Weight Matrices
//   /// @param Cols Col of Weight Matrices
//   NeuronMats(std::size_t Rows, std::size_t Cols);
  
//   public:
//   /// @brief Get the activated Value of Matrix
//   void activate();

//   /// @brief Get the derivative Value of Matrix
//   void derivative();

//   private:
//   /// @brief private Matrices
//   final_project::array2d<double> MatValue, MatActivateValue, MatDerivativeValue;
// }; // class Neuron Matrices


class Layer
{
  public:
  Layer();
  Layer(std::size_t size);

  Neuron & operator[](std::size_t index) { return Neurons[index]; }


  std::size_t getSize() const { return size; }


  private:
  std::size_t               size;
  std::unique_ptr<Neuron[]> Neurons;


}; // class Layer



class Network
{
  public:
  Network(std::vector<std::size_t> & topology);

  private:
  std::vector<std::size_t> topology;

  private:
  std::unique_ptr<Layer[]> Layers;
  std::unique_ptr<final_project::array2d<double>[]> weightMatrices;
  std::unique_ptr<final_project::array2d<double>[]> BiasMatrices;

  public:
  void show()
  {
    for (std::size_t i = 0; i < topology.size() - 1; ++i)
    {
      std::cout << weightMatrices[i] << std::endl;
      std::cout << BiasMatrices[i] << std::endl;
    }
  }

  public:
  void front_propagation(final_project::array2d<double> & input);
  void back_propagation(final_project::array2d<double> const & output, 
                        final_project::array2d<double> const & label);

  final_project::array2d<double> get_output() 
  {
    std::size_t output_layer_index = topology.size() - 1;
    Layer &output_layer = Layers[output_layer_index];
    final_project::array2d<double> output(1, topology.back());

    for (std::size_t i = 0; i < topology.back(); ++i) {
      output(0, i) = output_layer[i].get_activated_value();
    }

    return output;
  }  
};


// class NeuralNetwork
// {

//   public:
//   NeuralNetwork(std::vector<std::size_t> & topology);

//   private:
  
// };












} // namespace NeuralNetwork







/// Member Functions

#include <cmath>
#include <iostream>
#include <ranges>


namespace NeuralNetwork
{

// ---------------------------------------------------------------------------
// Class Neuron Member functions

/// @brief Default Constructor of Neuron Ojbect
/// @param value 
Neuron::Neuron()
: value {0} {
  activate();
  derivative();
}

/// @brief Constructor of Neuron Ojbect
/// @param value 
Neuron::Neuron(double value)
: value {value} {
  activate();
  derivative();
}

/// @brief Get the activated value
/// @details f(x) = \frac{x}{1 + abs(x)} 
void Neuron::activate()
  { activateValue = value / ( 1 + std::exp(value) ) ; }

/// @brief Get the derivative of input value
/// @details df(x) = x * (1 - x)
void Neuron::derivative() 
  { derivativeValue = value * ( 1 - value ); }

// ---------------------------------------------------------------------------
// Class Neuron Matrices Member Functions


// NeuronMats::NeuronMats(std::size_t Rows, std::size_t Cols)
//  : MatValue( Rows, Cols ), MatActivateValue( Rows, Cols ), MatDerivativeValue( Rows, Cols )
// {

//   MatValue.fill_random();
//   activate();
//   derivative();

//   std::cout << MatValue << std::endl;
//   std::cout << MatActivateValue << std::endl;
//   std::cout << MatDerivativeValue << std::endl;

// }


// void NeuronMats::activate()
// {

//   auto activate_func = [](auto & value){
//     auto activate_value { value / ( 1 + std::abs(value)) };
//     return activate_value;
//   };

//   std::transform(MatValue.begin(), MatValue.end(), MatActivateValue.begin(), activate_func);
// }

// void NeuronMats::derivative()
// {

//   auto derivative_func = [](auto & value){
//     auto derivative_value { value * ( 1 - value ) };
//     return derivative_value;
//   };

//   std::transform(MatValue.begin(), MatValue.end(), MatDerivativeValue.begin(), derivative_func);

// }


// ---------------------------------------------------------------------------
// Class Layer Member functions 

Layer::Layer()
: size {0}, Neurons {std::make_unique<Neuron[]>(0)} { }

/// @brief Constructor
/// @param size The number of neurons in this layer.
Layer::Layer(std::size_t size) 
: size {size}, Neurons {std::make_unique<Neuron[]>(size)} 
{
  for (std::size_t i = 0; i < size; ++i) { Neurons[i] = Neuron(0.0); }
}


// ---------------------------------------------------------------------------
// Class NeuralNetwork member functions
Network::Network(std::vector<std::size_t> & topology)
: topology {topology}
{
  Layers = std::make_unique<Layer[]>(topology.size());
  weightMatrices = std::make_unique<final_project::array2d<double>[]>(topology.size()  - 1);
  BiasMatrices = std::make_unique<final_project::array2d<double>[]>(topology.size()  - 1);

  for (std::size_t i = 0; i < topology.size(); ++i) { Layers[i] = Layer(topology[i]); }
  

  for (std::size_t i = 0; i < topology.size() - 1; ++i ) { 
    weightMatrices[i] = final_project::array2d<double>(topology[i], topology[i+1]); 
    weightMatrices[i].fill_random();

    BiasMatrices[i] = final_project::array2d<double>(1, topology[i+1]); 
    BiasMatrices[i].fill_random();
  }

}


/// @brief Do the front propagation 
/// @param input Input sample
void Network::front_propagation(final_project::array2d<double> & input)
  {
    for (std::size_t i = 0; i < input.rows(); ++i) 
      for (std::size_t j = 0; j < input.cols(); ++j)
        Layers[0][j].setValue(input(i,j));
    
    for (std::size_t i = 1; i < topology.size(); ++i)
    {
      Layer & prevLayer {Layers[i - 1]};
      Layer & currLayer {Layers[i]};


      for (std::size_t j = 0; j < currLayer.getSize(); ++j)
      {
        double sum {0.0};

        for (std::size_t k = 0; k < prevLayer.getSize(); ++k)
        {
          sum += prevLayer[k].get_activated_value() * weightMatrices[i-1](k,j);  
        }

        sum += BiasMatrices[i-1](0, j);
        currLayer[j].setValue(sum);
      }
    }
  }


/// @brief Do the back propagation
/// @param label The label of sample input
void Network::back_propagation(final_project::array2d<double> const & output, final_project::array2d<double> const & label)
{
  // Store the gradients of weights and biases
  std::vector<final_project::array2d<double>> weight_gradients(topology.size() - 1);
  std::vector<final_project::array2d<double>> bias_gradients(topology.size() - 1);

  for (std::size_t i = 0; i < topology.size() - 1; ++i) {
    weight_gradients[i] = final_project::array2d<double>(topology[i], topology[i+1]);
    bias_gradients[i] = final_project::array2d<double>(1, topology[i+1]);
  }

  // Compute output layer error
  Layer & outputLayer = Layers[topology.size() - 1];
  Layer & lastHiddenLayer = Layers[topology.size() - 2];
  final_project::array2d<double> output_errors(1, topology.back());

  for (std::size_t i = 0; i < topology.back(); ++i) {
    double error = label(0, i) - outputLayer[i].get_activated_value();
    output_errors(0, i) = error * outputLayer[i].get_derivative_Value();
  }

  // Backpropagate errors
  final_project::array2d<double> *current_errors = &output_errors;

  for (int i = topology.size() - 2; i >= 0; --i) {
    Layer & currLayer = Layers[i];
    Layer & nextLayer = Layers[i + 1];
    final_project::array2d<double> hidden_errors(1, topology[i]);

    for (std::size_t j = 0; j < topology[i + 1]; ++j) {
      bias_gradients[i](0, j) = (*current_errors)(0, j);//////////////////////////

      // for (std::size_t k = 0; k < topology[i]; ++k) {
      //   weight_gradients[i](k, j) += (*current_errors)(0, j) * currLayer[k].get_activated_value();
      // }
    }

  //   if (i > 0) {
  //     for (std::size_t j = 0; j < topology[i]; ++j) {
  //       double error = 0.0;

  //       for (std::size_t k = 0; k < topology[i + 1]; ++k) {
  //         error += (*current_errors)(0, k) * weightMatrices[i](j, k);
  //       }

  //       hidden_errors(0, j) = error * currLayer[j].get_derivative_Value();
  //     }

  //     current_errors = &hidden_errors;
  //   }
  // }

  // // Update weights and biases
  // for (std::size_t i = 0; i < topology.size() - 1; ++i) {
  //   for (std::size_t j = 0; j < topology[i]; ++j) {
  //     for (std::size_t k = 0; k < topology[i + 1]; ++k) {
  //       weightMatrices[i](j, k) += 0.1 * weight_gradients[i](j, k); // learning rate = 0.1
  //     }
  //   }

  //   for (std::size_t j = 0; j < topology[i + 1]; ++j) {
  //     BiasMatrices[i](0, j) += 0.1 * bias_gradients[i](0, j); // learning rate = 0.1
  //   }
  }
}



} // namespace NeuralNetwork






#endif /// _NEURON_CPP_