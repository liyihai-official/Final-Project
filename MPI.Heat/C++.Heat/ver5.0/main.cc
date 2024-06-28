#include <iostream>
#include "mpi_detials/mpi_topology.hpp"
#include "mpi_detials/mpi_types.hpp"
#include "mpi_detials/mpi_environment.hpp"
#include "multi_array/base.hpp"
#include "mpi_distribute/mpi_distribute_array.hpp"
#include "array.hpp"

#include <cmath>


// template <typename T, std::size_t NumDims>
// struct heat_equation {

//   heat_equation(final_project::__detail::__types::__multi_array_shape<NumDims> global_grid_shape)
//   {
//     for (std::size_t i = 0; i < NumDims; ++i)
//     {
//       minRange[i] = 0;
//       maxRange[i] = 1;
//       deltaXs[i] = (maxRange[i] - minRange[i]) / global_grid_shape[i];

//       dt = 1.0 / std::pow(2, NumDims);
//       dt = std::min(dt, 0.1);

//       weights[i] = coff * dt / (deltaXs[i] * deltaXs[i]);
//       diags[i] = -2.0 + (deltaXs[i] * deltaXs[i]) / (NumDims * coff * dt);
//     }
//   }

//   private:
//   T coff {1};
//   T dt {0.1}; 
//   T diags[NumDims], weights[NumDims], minRange[NumDims], maxRange[NumDims], deltaXs[NumDims];

// };

int main( int argc, char ** argv)
{
  auto world {final_project::mpi::env(argc, argv)};


  auto shape {final_project::__detail::__types::__multi_array_shape<2>(15, 17)};
  // auto an_topology {final_project::__detail::__mpi_types::__mpi_topology<double, 2>(shape, world)};
  // std::cout 
  // << " PROCESS " << an_topology.__rank 
  // << " Has Coordinate : \t ["
  // << an_topology.__coordinates[0] << ", "
  // << an_topology.__coordinates[1] << ", "
  // << an_topology.__coordinates[2] << "]"
  // << " \t "
  // << " Has Shape : \t [" 
  // << an_topology.__local_shape[0] << ", "
  // << an_topology.__local_shape[1] << ", "
  // << an_topology.__local_shape[2] << "] "
  // << " Range : \t [ ("
  // << an_topology.__starts[0] << ", " << an_topology.__ends[0] << ") " << ", ("
  // << an_topology.__starts[1] << ", " << an_topology.__ends[1] << ") " << ", ("
  // << an_topology.__starts[2] << ", " << an_topology.__ends[2] << ") " << "] "
  // << std::endl;


  // auto Array {final_project::__detail::__multi_array::__array<float, 2>(shape)};
  // Array.fill(0);

  // if (world.rank() == 0 )
  //   std::cout << Array << std::endl;
  // }


  auto DA {final_project::__detail::__mpi_distribute_array<double, 2>(shape, world)};
  DA.__fill_boundary(1);

  // if (world.rank() == 0) 
  // {
  // std::cout << DA << std::endl;
  // }

  // heat_equation<double, 2> equation {shape};
  if (world.rank() == 0)
  { }

  // 
  auto DD {final_project::array::array_distribute<double, 2>(shape, world)};



  std::cout << *(DD.body.get()) << std::endl;



  return 0;
}