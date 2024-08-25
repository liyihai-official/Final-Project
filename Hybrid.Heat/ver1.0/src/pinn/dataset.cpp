///
/// @file dataset.cpp
/// @brief Generate Random dataset for training.
///


#ifndef FINAL_PROJECT_DATASET_CPP
#define FINAL_PROJECT_DATASET_CPP


#include <pinn/dataset.hpp>


#include <vector>

#ifndef NDEBUG
  #include <iostream>
  #include <iomanip>
#endif 


namespace final_project { namespace PINN {

void dataset::save(  
  const String dataset_X_internal,
  const String dataset_X_boundary,
  const String dataset_Y_boundary)
{
  torch::save(X_boundary, dataset_X_boundary);
  torch::save(Y_boundary, dataset_Y_boundary);
  torch::save(X_internal, dataset_X_internal);
}

dataset& dataset::operator=( dataset && other) noexcept
{
  if (this != &other)
  {
    X_boundary = std::move(other.X_boundary);
    X_internal = std::move(other.X_internal);
    Y_boundary = std::move(other.Y_boundary);
  }
  return *this;
}

dataset::dataset(
  const Integer in_size /*IN_SIZE*/,  const Integer out_size  /*OUT_SIZE*/,
  const Integer nx      /*NX*/,       const Integer ny        /*NY*/,
  const torch::Device & device)
: in_size {in_size}, out_size {out_size}, 
  nx {nx}, ny {ny}, 
  X_boundary {nullptr}, Y_boundary {nullptr}
{
  std::vector<value_type> internal_Xobj (nx*2*ny);  
  for (size_type x = 0; x < nx; ++x)
  {
    for (size_type y = 0; y < ny; ++y)
    {
      size_type idx_base { 2 * (x * ny + y) };
      internal_Xobj[idx_base]   = x * (1.0  / (nx - 1));
      internal_Xobj[idx_base+1] = y * (1.0  / (ny - 1));
    }
  }

  X_internal = torch::from_blob(
    internal_Xobj.data(), 
    {nx * ny, in_size},
    torch::requires_grad()
  ).to(device);
}

dataset::dataset(
  const Integer nx, const Integer ny,
  const String dataset_X_internal,
  const String dataset_X_boundary,
  const String dataset_Y_boundary, const torch::Device & device)
{
  FINAL_PROJECT_ASSERT_MSG(
       std::filesystem::exists(dataset_X_boundary) 
    && std::filesystem::exists(dataset_X_internal) 
    && std::filesystem::exists(dataset_Y_boundary), "No Existing Datasets.");
  
  torch::load(X_internal, dataset_X_internal);
  torch::load(X_boundary, dataset_X_boundary);
  torch::load(Y_boundary, dataset_Y_boundary);

  in_size = X_boundary.size(1);
  out_size = Y_boundary.size(1);
  
  FINAL_PROJECT_ASSERT_MSG(
    (
      ((nx-2+ny-2)*2      == X_boundary.size(0))
   && ((nx-2)*(ny-2)  == X_internal.size(0))
   && (X_boundary.size(0) == Y_boundary.size(0))
    ), "Invalid Dataset, does not match given grid sizes."
  );

  X_boundary = X_boundary.to(device);
  Y_boundary = Y_boundary.to(device);
  X_internal = X_internal.to(device);
}


dataset::dataset(
  const Integer in_size, const Integer out_size, 
  const Integer nx, const Integer ny,
  std::mt19937 & rde,
  std::uniform_real_distribution<value_type> & rng,
  BCFunction & FuncDim00, BCFunction & FuncDim01, BCFunction & FuncDim10, BCFunction &  FuncDim11,
  const torch::Device & device)
: in_size {in_size}, out_size {out_size}, nx {nx}, ny {ny}
{
  std::vector<value_type> internal_Xobj, boundary_Xobj, boundary_Yobj;
  
  /// internal_Xobj
  for (size_type x = 1; x < nx-1; ++x)
  {
    for (size_type y = 1; y < ny-1; ++y)
    {
      internal_Xobj.push_back(rng(rde));
      internal_Xobj.push_back(rng(rde));
    }
  }


  /// boundary_Xobj, boundary_Yobj
  /// Dimension 0
  for (size_type y = 1; y < ny-1; ++y)
  {
    auto num {rng(rde)};
    boundary_Xobj.push_back(0.0); boundary_Xobj.push_back(num); //  (0, y)
    boundary_Yobj.push_back(FuncDim00(0.0, num));               // g(0, y)

    num = rng(rde);
    boundary_Xobj.push_back(1.0); boundary_Xobj.push_back(num); //  (1, y)
    boundary_Yobj.push_back(FuncDim01(1.0, num));               // g(1, y)
  }

  /// Dimension 1
  for (size_type x = 1; x < nx-1; ++x)
  {
    auto num {rng(rde)};
    boundary_Xobj.push_back(num); boundary_Xobj.push_back(0.0); //  (x, 0)
    boundary_Yobj.push_back(FuncDim10(num, 0.0));               // g(x, 0)

    num = rng(rde);
    boundary_Xobj.push_back(num); boundary_Xobj.push_back(1.0); //  (x, 1)
    boundary_Yobj.push_back(FuncDim11(num, 1.0));               // g(x, 1)
  }

  // To Torch::Tensor to(device)
  size_type num_rows = internal_Xobj.size() / static_cast<size_type>(in_size);
  X_internal = torch::from_blob(internal_Xobj.data(), 
    {num_rows, in_size}, torch::requires_grad()  ).to(device);

  num_rows = boundary_Xobj.size() / static_cast<size_type>(in_size);
  X_boundary = torch::from_blob(boundary_Xobj.data(),
    {num_rows,  in_size}                         ).to(device);

  num_rows = boundary_Yobj.size() / static_cast<size_type>(out_size);
  Y_boundary = torch::from_blob(boundary_Yobj.data(),
    {num_rows, out_size}                         ).to(device);
}



dataset::dataset(
  const Integer in_size /*IN_SIZE*/,  const Integer out_size /*OUT_SIZE*/,
  const Integer nx /*NX*/,            const Integer ny /*NY*/,      const Integer nz /*NZ*/,
  const torch::Device & device)
: in_size {in_size}, out_size {out_size}, 
  nx {nx}, ny {ny}, nz {nz},
  X_boundary {nullptr}, Y_boundary {nullptr}
{
  std::vector<value_type> internal_Xobj_full (nx*nz*ny*3);  
  for (size_type x = 0; x < nx; ++x)
  {
    for (size_type y = 0; y < ny; ++y)
    {
      for (size_type z = 0; z < nz; ++z)
      {
        size_type idx_base { 3 * (x * ny * nz + y * nz + z) };
        internal_Xobj_full[idx_base]   = x * (1.0  / (nx - 1));
        internal_Xobj_full[idx_base+1] = y * (1.0  / (ny - 1));
        internal_Xobj_full[idx_base+2] = z * (1.0  / (nz - 1));
      }
    }
  }

  X_internal = torch::from_blob(internal_Xobj_full.data(), 
    {nx * ny * nz, in_size}, torch::requires_grad()).to(device);
}


dataset::dataset(
  const Integer nx, const Integer ny, const Integer nz,
  const String dataset_X_internal,
  const String dataset_X_boundary,
  const String dataset_Y_boundary, const torch::Device & device)
{
  FINAL_PROJECT_ASSERT_MSG(
       std::filesystem::exists(dataset_X_boundary) 
    && std::filesystem::exists(dataset_X_internal) 
    && std::filesystem::exists(dataset_Y_boundary), "No Existing Datasets.");
  
  torch::load(X_internal, dataset_X_internal);
  torch::load(X_boundary, dataset_X_boundary);
  torch::load(Y_boundary, dataset_Y_boundary);

  in_size   = X_boundary.size(1);
  out_size  = Y_boundary.size(1);

  auto boundary_num { ((nx-2)*(ny-2) + (ny-2)*(nz-2) + (nz-2)*(nx-2)) * 2 };
  FINAL_PROJECT_ASSERT_MSG(
    (
      (boundary_num           == X_boundary.size(0))
   && ((nx-2)*(ny-2)*(nz-2)   == X_internal.size(0))
   && (X_boundary.size(0)     == Y_boundary.size(0))
    ), "Invalid Dataset, does not match given grid sizes."
  );

  X_boundary = X_boundary.to(device);
  Y_boundary = Y_boundary.to(device);
  X_internal = X_internal.to(device);
}


dataset::dataset(
  const Integer in_size, const Integer out_size, 
  const Integer nx, const Integer ny, const Integer nz,
  std::mt19937 & rde,
  std::uniform_real_distribution<value_type> & rng,
  BCFunction3D & FuncDim00, BCFunction3D &  FuncDim01, 
  BCFunction3D & FuncDim10, BCFunction3D &  FuncDim11, 
  BCFunction3D & FuncDim20, BCFunction3D &  FuncDim21,
  const torch::Device & device)
: in_size {in_size}, out_size {out_size}, nx {nx}, ny {ny}, nz {nz}
{
  std::vector<value_type> internal_Xobj, boundary_Xobj, boundary_Yobj;
  
  /// internal_Xobj 
  for (size_type x = 1; x < nx-1; ++x)
  {
    for (size_type y = 1; y < ny-1; ++y)
    {
      for (size_type z = 1; z < nz-1; ++z)
      {
        internal_Xobj.push_back(rng(rde)); internal_Xobj.push_back(rng(rde)); internal_Xobj.push_back(rng(rde)); // (x, y, z)
      }
    }
  }

  /// boundary_Xobj, boundary_Yobj
  /// Dimension 0
  for (size_type y = 1; y < ny-1; ++y)
  {
    for (size_type z = 1; z < nz-1; ++z)
    {
      auto num_0 {rng(rde)}; auto num_1 {rng(rde)};
      boundary_Xobj.push_back(0.0); boundary_Xobj.push_back(num_0); boundary_Xobj.push_back(num_1); //  (0, y, z)
      boundary_Yobj.push_back(FuncDim00(0.0, num_0, num_1));                                        // g(0, y, z)

      num_0 = rng(rde);           num_1 = rng(rde);
      boundary_Xobj.push_back(1.0); boundary_Xobj.push_back(num_0); boundary_Xobj.push_back(num_1); //  (1, y, z)
      boundary_Yobj.push_back(FuncDim01(1.0, num_0, num_1));                                        // g(1, y, z)
    }
  }

  /// Dimension 1
  for (size_type x = 1; x < nx-1; ++x)
  {
    for (size_type z = 1; z < nz-1; ++z)
    {
      auto num_0 {rng(rde)}; auto num_1 {rng(rde)};
      boundary_Xobj.push_back(num_0); boundary_Xobj.push_back(0.0); boundary_Xobj.push_back(num_1); //  (x, 0, z)
      boundary_Yobj.push_back(FuncDim10(num_0, 0.0, num_1));                                        // g(x, 0, z)

      num_0 = rng(rde);           num_1 = rng(rde);
      boundary_Xobj.push_back(num_0); boundary_Xobj.push_back(1.0); boundary_Xobj.push_back(num_1); //  (x, 1, z)
      boundary_Yobj.push_back(FuncDim11(num_0, 1.0, num_1));                                        // g(x, 1, z)
    }
  }

  /// Dimension 2
  for (size_type x = 1; x < nx-1; ++x)
  {
    for (size_type y = 1; y < ny-1; ++y)
    {
      auto num_0 {rng(rde)}; auto num_1 {rng(rde)};
      boundary_Xobj.push_back(num_0); boundary_Xobj.push_back(num_1); boundary_Xobj.push_back(0.0); //  (x, y, 0)
      boundary_Yobj.push_back(FuncDim20(num_0, num_1, 0.0));                                        // g(x, y, 0)

      num_0 = rng(rde);           num_1 = rng(rde);
      boundary_Xobj.push_back(num_0); boundary_Xobj.push_back(num_1); boundary_Xobj.push_back(1.0); //  (x, y, 1)
      boundary_Yobj.push_back(FuncDim21(num_0, num_1, 1.0));                                        // g(x, y, 1)
    }
  }


  // To Torch::Tensor to(device)
  size_type num_rows = internal_Xobj.size() / static_cast<size_type>(in_size);

  X_internal = torch::from_blob(internal_Xobj.data(), 
    {num_rows, in_size}, torch::requires_grad()  ).to(device);

  num_rows = boundary_Xobj.size() / static_cast<size_type>(in_size);
  X_boundary = torch::from_blob(boundary_Xobj.data(),
    {num_rows,  in_size}                         ).to(device);

  num_rows = boundary_Yobj.size() / static_cast<size_type>(out_size);
  Y_boundary = torch::from_blob(boundary_Yobj.data(),
    {num_rows, out_size}                         ).to(device);
}




}}



#endif // end define FINAL_PROJECT_DATASET_CPP