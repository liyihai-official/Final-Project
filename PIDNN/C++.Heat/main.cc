#include <iostream>

#include "neuron.cpp"
#include "../../MPI.Heat/C++.Heat/ver3.0/multi_array/array_distribute.cpp"
#include "../../MPI.Heat/C++.Heat/ver3.0/multi_array/array_io.cpp"

int main ( int argc, char ** argv)
{

  std::unique_ptr<NeuralNetwork::Neuron> N1, N2, N3;

  N1 = std::make_unique<NeuralNetwork::Neuron>(0.2);
  N2 = std::make_unique<NeuralNetwork::Neuron>(0.3);
  N3 = std::make_unique<NeuralNetwork::Neuron>(0.4);


  std::cout
  << " Value : " << N1->getVal() 
  << " Activate Value : " << N1->get_activated_value() 
  << " Derivate " << N1->get_derivative_Value() << std::endl;

  std::cout 
  << " Value : " << N2->getVal() 
  << " Activate Value : " << N2->get_activated_value() 
  << " Derivate " << N2->get_derivative_Value() << std::endl;

  std::cout 
  << " Value : " << N3->getVal() 
  << " Activate Value : " << N3->get_activated_value() 
  << " Derivate " << N3->get_derivative_Value() << std::endl;

  

  auto Layer {NeuralNetwork::Layer(3)};



  auto A = final_project::array2d<double>(3,3);
  std::cout << " ------------------------------------- \n";
  A.fill_random();

  std::cout << A << std::endl;

  A.transpose();
  
  std::cout << A << std::endl;



  std::cout << " ------------------------------------- \n";
  std::vector<std::size_t> topology {3, 2, 3};

  // for (auto const & t : topology) { std::cout << t << " "; }
  // std::cout << "\n";

  NeuralNetwork::Network nn (topology);

  nn.show();

  std::cout << " ------------------------------------- \n";
  auto input = final_project::array2d<double>(1,3);
  input.fill_random();
  std::cout << input << std::endl;

  std::cout << " ------------------------------------- \n";
  nn.front_propagation(input);

  




  return 0;
}