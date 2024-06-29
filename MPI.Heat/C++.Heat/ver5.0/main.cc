#include <iostream>
#include "mpi_detials/mpi_topology.hpp"
#include "mpi_detials/mpi_types.hpp"
#include "mpi_detials/mpi_environment.hpp"
#include "multi_array/base.hpp"
#include "mpi_distribute/mpi_distribute_array.hpp"
#include "array.hpp"

#include <cmath>
#include <omp.h>


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

template <typename T, std::size_t NumDim>
void Gather(final_project::array::array_base<T, NumDim>       & gather,   
            final_project::array::array_distribute<T, NumDim> & array)
{
  int root {0};
  int pid, i, j, k;

  std::size_t indexs[NumDim];

  MPI_Datatype sbuf_block, temp, mpi_T {final_project::__detail::__mpi_types::__get_mpi_type<T>()};

  int num_procs {array.get_topology().__num_procs};

  int Ns[NumDim], starts_cpy[NumDim];

  int s_list[NumDim][num_procs], n_list[NumDim][num_procs];

  for (std::size_t dim = 0; dim < NumDim; ++dim)
  {
    indexs[dim] = 1;
    Ns[dim] = array.get_topology().__ends[0] - array.get_topology().__starts[0] + 1;
    starts_cpy[dim] = array.get_topology().__starts[0];

    if (starts_cpy[dim] == 1) 
    {
      -- starts_cpy[dim];
      -- indexs[dim];
      ++ Ns[dim];
    }

    if (array.get_topology().__ends[0] == array.get_topology().__global_shape[dim] - 2) 
      ++ Ns[dim];


    MPI_Gather(&starts_cpy[dim], 1, MPI_INT, s_list[dim], 1, MPI_INT, root, array.get_topology().__comm_cart);
    MPI_Gather(&Ns[dim], 1, MPI_INT, n_list[dim], 1, MPI_INT, root, array.get_topology().__comm_cart);
  }


  if (array.get_topology().__rank == root);
  {


    for (pid = 0; pid < num_procs; ++pid)
    {
      if (pid == root)
      {
        for ( i = starts_cpy[pid]; i <= array.get_topology().__ends[0]; ++i)
        {
          // for (j = starts)
        }
      }
    }
  }


}

int main( int argc, char ** argv)
{
  auto world {final_project::mpi::env(argc, argv)};


  auto shape {final_project::__detail::__types::__multi_array_shape<2>(9, 13)};
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


  // auto DA {final_project::__detail::__mpi_distribute_array<double, 2>(shape, world)};
  // DA.__fill_boundary(1);

  // if (world.rank() == 0) 
  // {
  // std::cout << DA << std::endl;
  // }

  // heat_equation<double, 2> equation {shape};
  if (world.rank() == 0)
  { }

  // 
  auto DD {final_project::array::array_distribute<double, 2>(shape, world)};
  DD.fill_boundary(1);

  // std::cout << DD.get_array() << std::endl;
  double diff {0}, gdiff {0};

  auto t1 = MPI_Wtime();
  for (std::size_t i = 0; i < 100; ++i)
  {
    diff = DD.update();
    MPI_Reduce(&diff, &gdiff, 1, MPI_DOUBLE, MPI_SUM, 0, world.comm());
    if (world.rank() == 0 && i % 10 == 0) std::cout << std::fixed << std::setprecision(15) << std::setw(15) << gdiff << std::endl;
  }
  auto t2 = MPI_Wtime();
  std::cout << DD.get_array() << std::endl;

  MPI_Barrier(world.comm());
  auto G {final_project::array::array_base<double, 2>(shape)};
  if (world.rank() == 0) std::cout << G.get_array() << std::endl;

  // std::cout << t2 - t1 << std::endl;



  return 0;
}