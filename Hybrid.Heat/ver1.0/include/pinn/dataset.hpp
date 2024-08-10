///
/// @file dataset.hpp
/// @brief Generate Random dataset for training.
///


#ifndef FINAL_PROJECT_DATASET_HPP
#define FINAL_PROJECT_DATASET_HPP

#pragma once 
#include <torch/torch.h>

///

#include <types.hpp>
#include <multiarray.hpp>

///



namespace final_project { namespace PINN {


struct dataset
{

  dataset() = default;
  dataset(const Integer /*IN_SIZE*/,  const Integer /*OUT_SIZE*/, 
          const Integer /*NX*/,       const Integer /*NY*/, 
          std::mt19937 & /* Random Number Generator */,
          std::uniform_real_distribution<value_type> & /* Distribution */,
          const torch::Device &);

  dataset(const Integer /*IN_SIZE*/,  const Integer /*OUT_SIZE*/,
          const Integer /*NX*/,       const Integer /*NY*/,
          const torch::Device &);

#ifndef NDEBUG

  void show_X_internal();
  void show_X_boundary();
  void show_Y_boundary();

#endif 


  torch::Tensor X_internal, X_boundary, Y_boundary;
  Integer in_size, out_size, nx, ny;
}; // dataset



} // namespace PINN
} // namespace final_project  


///
///
/// --------------------------- Inline Function Definitions  ---------------------------  ///
///
///

#include <random>
#include <vector>

#ifndef NDEBUG
  #include <iostream>
  #include <iomanip>
#endif 


namespace final_project { namespace PINN {

typedef Float value_type;

inline dataset::dataset(
  const Integer in_size, const Integer out_size, 
  const Integer nx, const Integer ny,
  std::mt19937 & rde,
  std::uniform_real_distribution<value_type> & rng,
  const torch::Device & device)
: in_size {in_size}, out_size {out_size}, nx {nx}, ny {ny}
{

  std::vector<value_type> internal_Xobj, boundary_Xobj;

  for (size_type x = 0; x < nx; ++x)
  {
    for (size_type y = 0; y < ny; ++y)
    {
      internal_Xobj.push_back(rng(rde));
      internal_Xobj.push_back(rng(rde));
    }
  }

  for (size_type y = 0; y < ny; ++y)
  {
    boundary_Xobj.push_back(0.0);
    boundary_Xobj.push_back(rng(rde));

    boundary_Xobj.push_back(1.0);
    boundary_Xobj.push_back(rng(rde));
  }


  for (size_type x = 0; x < nx; ++x)
  {
    boundary_Xobj.push_back(rng(rde));
    boundary_Xobj.push_back(0.0);

    boundary_Xobj.push_back(rng(rde));
    boundary_Xobj.push_back(1.0);
  }


  X_internal = torch::from_blob(
    internal_Xobj.data(), 
    {nx * ny, in_size},
    torch::requires_grad()
  ).to(device);

  X_boundary = torch::from_blob(
    boundary_Xobj.data(),
    {2 * (nx + ny), in_size}
  ).to(device);

  Y_boundary = torch::ones({2 * (nx + ny),  out_size}, device);
}


inline dataset::dataset(
  const Integer in_size /*IN_SIZE*/,  const Integer out_size/*OUT_SIZE*/,
  const Integer nx /*NX*/,       const Integer ny /*NY*/,
  const torch::Device & device)
: in_size {in_size}, out_size {out_size}, 
  nx {nx}, ny {ny}, 
  X_boundary {nullptr}, Y_boundary {nullptr}
{
  multi_array::array_base<value_type, 2> internal_Xobj (nx, 2*ny);  

  for (size_type x = 0; x < nx; ++x)
  {
    for (size_type y = 0; y < 2*ny; y+=2)
    {
      internal_Xobj(x, y)   = x * (1.0        / (nx - 1));
      internal_Xobj(x, y+1) = y * (1.0 / 2.0  / (ny - 1));
    }
  }

  X_internal = torch::from_blob(
    &internal_Xobj.data(), 
    {nx * ny, in_size},
    torch::requires_grad()
  ).to(device);
}


#ifndef NDEBUG

inline void dataset::show_X_internal()
{
  std::cout 
    << "----------------------------------- GRID INTERNAL -----------------------------------"  << "\n"
    << "X_internal sizes (grid): "     << X_internal.sizes()                    << "\n"
    << "X_internal.device().index(): " << X_internal.device().index()           << "\n"
    << "X_internal.requires_grad(): "  << X_internal.requires_grad()            << "\n"
    << std::endl;

  // for (size_type x = 0; x < NX; ++x)
  // {
  //   std::cout << std::fixed << std::setw(3) << x;
  //   for (size_type y = 0; y < 2 * NY; y+=2)
  //   {
  //     auto index {x * NY + y};
  //     std::cout 
  //       << std::fixed << std::setprecision(2) << std::setw(4)
  //       << "["  << X_data[index] 
  //       << ", " << X_data[index + 1]
  //       << "]  ";
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << std::endl;

}

inline void dataset::show_X_boundary()
{
  std::cout 
    << "---------------------------------- GRID BOUNDARY -----------------------------------"  << "\n"
    << "X_boundary sizes (grid): "       << X_boundary.sizes()          << "\n"
    << "X_boundary.device().index(): "   << X_boundary.device().index() << "\n"
    << "X_boundary.requires_grad(): "    << X_boundary.requires_grad()  << "\n"
    << std::endl;

  // for (size_type idx = 0; idx < 2 * 2 * (NX + NY); idx+=2)
  // {
  //     std::cout 
  //       << std::fixed << std::setw(3) << idx / 2
  //       << std::fixed << std::setprecision(3) << std::setw(4)
  //       << "["  << X_train_data[idx] 
  //       << ", " << X_train_data[idx+1]
  //       << "]  "
  //       << std::endl;
  // }
  // std::cout << std::endl;

}

inline void dataset::show_Y_boundary()
{
  std::cout 
    << "------------------------------------- Y Boundary ------------------------------------- "  << "\n"
    << "Y_boundary sizes: "           << Y_boundary.sizes()         << "\n"
    << "Y_boundary.device().type(): " << Y_boundary.device().type() << "\n"
    << "Y_boundary.requires_grad(): " << Y_boundary.requires_grad() << "\n"
    << std::endl;
}

#endif 











} // namespace PINN
} // namespace final_project  


#endif // end define FINAL_PROJECT_DATASET_HPP