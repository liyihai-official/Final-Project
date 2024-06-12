#ifndef _NEURON_CPP_
#define _NEURON_CPP_

#pragma once 
#include <iostream>
#include <cmath>

class Neuron
{
  public:
  Neuron(double value);

  void activate();
  void derivative();

  public:
  double getVal() {  return this->value;  }

  double get_activated_value() { return this->activateValue; }
  double get_derivative_Value() { return this->derivativeValue; }

  private:
  double value, activateValue, derivativeValue;
};


Neuron::Neuron(double value)
: value {value} 
{
  activate();
  derivative();
}

void Neuron::activate()
{
  activateValue = value / ( 1 + std::abs(value) ) ;
}

void Neuron::derivative()
{
  derivativeValue = value * ( 1 - value );
}


#endif /// _NEURON_CPP_