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
  double getVal() {  return this->value;  }

  double get_activated_value() { return this->activateValue; }
  double get_derivative_Value() { return this->derivativeValue; }

  private:
  double value, activateValue, derivativeValue;
}; // class Neuron

class NeuronMats
{
  public:
  NeuronMats();
  NeuronMats(std::size_t Rows, std::size_t Cols);
  
  private:
  final_project::array2d<double> MatValue, MatActivateValue, MatDerivativeValue;
}; // class Neuron Matrices


class Layer
{

  public:
  Layer();
  Layer(std::size_t size);

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

  public:
  void show()
  {
    for (std::size_t i = 0; i < topology.size() - 1; ++i)
    {
      std::cout << weightMatrices[i] << std::endl;
    }
  }

  public:
  void front_propagation(final_project::array2d<double> & input)
  {
FINAL_PROJECT_ASSERT_MSG(input.Cols == weightMatrices[0].Rows, "Invalid shape of input sample.");
    for (std::size_t i = 0; i < topology.size() - 1; ++i)
    {
      input = input * weightMatrices[i];
    }
    std::cout << input << std::endl;
  }
};

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
void Neuron::activate()
  { activateValue = value / ( 1 + std::abs(value) ) ; }

/// @brief Get the derivative of input value
void Neuron::derivative() 
  { derivativeValue = value * ( 1 - value ); }

// ---------------------------------------------------------------------------
// Class Neuron Matrices Member Functions
// NeuronMats::NeuronMats(std::size_t Rows, std::size_t Cols)
// {

// }



// ---------------------------------------------------------------------------
// Class Layer Member functions 

Layer::Layer()
: size {0}, Neurons {std::make_unique<Neuron[]>(0)} { }

/// @brief Constructor
/// @param size The number of neurons in this layer.
Layer::Layer(std::size_t size) 
: size {size}, Neurons {std::make_unique<Neuron[]>(size)} {
  for (std::size_t i = 0; i < size; ++i) { Neurons[i] = Neuron(0.0); }

}


// ---------------------------------------------------------------------------
// Class NeuralNetwork member functions
Network::Network(std::vector<std::size_t> & topology)
: topology {topology}
{
  Layers = std::make_unique<Layer[]>(topology.size());
  weightMatrices = std::make_unique<final_project::array2d<double>[]>(topology.size()  - 1);

  for (std::size_t i = 0; i < topology.size(); ++i) { Layers[i] = Layer(topology[i]); }

  for (std::size_t i = 0; i < topology.size() - 1; ++i ) { 
    weightMatrices[i] = final_project::array2d<double>(topology[i], topology[i+1]); 
    weightMatrices[i].fill_random();
  }
    
}

} // namespace NeuralNetwork






#endif /// _NEURON_CPP_