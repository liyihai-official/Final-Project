/**
 * @file main.hpp
 * 
 * @brief This file includes all necessary headers for the Heat Equation solver.
 * 
 * This header file is used to include the various modules required for the 
 * parallel processing of the Heat Equation. The modules include environment 
 * setup, array operations, sweeping operations, data exchange, data gathering, 
 * and initialization.
 * 
 * @author Li Yihai
 * @version 3.0
 * @date May 25, 2024
 * 
 * @section DESCRIPTION
 * This header file consolidates the necessary includes for the Heat Equation 
 * solver, which is implemented using a distributed approach. The included 
 * modules collectively enable the setup, computation, and finalization of 
 * the heat equation solution in a parallelized environment.
 */

#ifndef FINAL_PROJECT_HPP_LIYIHAI
#define FINAL_PROJECT_HPP_LIYIHAI

#include "environment.cpp"

#include "array.cpp"
#include "sweep.cpp"
#include "exchange.cpp"
#include "gather.cpp"

#include "initialization.cpp"

#endif