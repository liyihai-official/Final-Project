
#include "final_project.cpp"
#include "heat.cpp"

#include <iostream>
#include <iomanip>
#include <mpi.h>

int main ( int argc, char ** argv )
{

  auto world {final_project::mpi::env(argc, argv)};


  auto A {final_project::array::array_distribute<double, 2>(world, 7, 9)};
  auto B {final_project::array::array_distribute<double, 2>(world, 7, 9)};

  // std::cout << A(1,1) << "\n";

  if (world.rank() == 0)
  {
    std::cout << B << std::endl;
  } 

  auto t1 = MPI_Wtime();
  for ( int i = 0; i < 1000; ++i)
  {
    
  }
  auto t2 = MPI_Wtime();

  auto C = final_project::heat_equation::_heat_pure_mpi<double, 2>(world, 7, 9);
  auto D = final_project::heat_equation::_heat_pure_mpi<double, 2>(world, 7, 9);

  auto E = final_project::heat_equation::_heat_2d<double>(world, 7, 9);
  auto F = final_project::heat_equation::_heat_2d<double>(world, 7, 9);
  // C._sweep(D);
  // std::cout << C._grid_world(1,1);
  E._sweep(F);

  if (world.rank() == 0)
  {
    std::cout << F._grid_world << std::endl;
  } 







  return 0;
}