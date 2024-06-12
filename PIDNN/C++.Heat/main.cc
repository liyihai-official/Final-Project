#include <iostream>
#include <memory>

#include "neuron.cpp"

int main ( int argc, char ** argv)
{

  std::unique_ptr<Neuron> N1, N2, N3;

  N1 = std::make_unique<Neuron>(0.2);
  N2 = std::make_unique<Neuron>(0.3);
  N3 = std::make_unique<Neuron>(0.4);


  std::cout 
  << "Value : " << N1->getVal() 
  << " Activate Value : " << N1->get_activated_value() 
  << " Derivate " << N1->get_derivative_Value() << std::endl;

  std::cout 
  << "Value : " << N2->getVal() 
  << " Activate Value : " << N2->get_activated_value() 
  << " Derivate " << N2->get_derivative_Value() << std::endl;

  std::cout 
  << "Value : " << N3->getVal() 
  << " Activate Value : " << N3->get_activated_value() 
  << " Derivate " << N3->get_derivative_Value() << std::endl;

  







  return 0;
}