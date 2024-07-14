#include <iostream>
#include <cmath>

// #include <omp.h>
#include <vector>
#include <cstring>

#include "fdm/evolve.hpp"
#include "fdm/heat.hpp"


int main( int argc, char ** argv)
{

  constexpr double tol {1E-4};
  constexpr std::size_t nsteps {1000000}, stepinterval {nsteps / 100};
  constexpr std::size_t numDIM {2}, nx {1000}, ny {1000};

  bool converge {false};
  std::size_t iter {0};  
  double gdiff {0.0}, ldiff {0.0}, ttime {0.0};

  auto mpi_world  {final_project::mpi::env(argc, argv)};
  auto glob_shape {final_project::__detail::__types::__multi_array_shape<numDIM>(nx, ny)};

  // Setups 
  auto heat_equation {final_project::heat_equation<double, numDIM>(glob_shape)};

  auto gather {final_project::array::array_base<double,2>(glob_shape)};
  auto ping {final_project::array::array_distribute<double, numDIM>(glob_shape, mpi_world)};
  auto pong {final_project::array::array_distribute<double, numDIM>(glob_shape, mpi_world)};
  

  // Brief informations
  MPI_Barrier(mpi_world.comm());
  if (0 == mpi_world.rank())
  {
    std::cout << "Heat " << numDIM << "D Simulation Parameters: " << std::endl;;
    std::cout << "\tRows: " << nx << " Columns: " << ny << std::endl;;
    std::cout << "\tTime steps: " << nsteps << std::endl;
    std::cout << "MPI Parameters: " << std::endl;
    std::cout << "\tNumber of MPI Processes: " << mpi_world.size() << std::endl;
    std::cout << "\tRoot Process: " << 0 << std::endl;


    std::cout << "Heat Parameters: "    << std::endl;
    std::cout << "\tCoefficient: "      << heat_equation.coff << "\n"
              << "\tTime resolution: "  << heat_equation.dt   << "\n"
              << "\tWeights: "          << heat_equation.weights[0] << ", " << heat_equation.weights[1] << "\n"
              << "\tdxs: "              << heat_equation.dxs[0]     << ", " << heat_equation.dxs[1]     << std::endl;
  }

  // setups
  ping.fill_boundary(10);
  pong.fill_boundary(10);

  MPI_Barrier(mpi_world.comm());
  sleep(1);
  MPI_Barrier(mpi_world.comm());


  // Time Evolve
  auto start_clock {MPI_Wtime()};
  for (iter = 0; iter < nsteps; ++iter)
  {
    exchange_ping_pong1(ping);
    ldiff = update_ping_pong1(ping, pong, heat_equation);
    MPI_Allreduce(&ldiff, &gdiff, 1, MPI_DOUBLE, MPI_SUM, mpi_world.comm());

    if (mpi_world.rank() == 0 && iter % stepinterval == 0) 
      std::cout << std::fixed << std::setprecision(13) << std::setw(15) << gdiff << "\t" << ldiff << std::endl;

    if (gdiff  <= tol) {
      std::cout << "Converge at : " << std::fixed << std::setw(7) << iter << std::endl;
      converge = true;
      break;
    }
    
    ping.swap(pong);
  }
  auto stop_clock {MPI_Wtime()-start_clock};

  // results
  if (converge)
  {
    Gather(gather, pong);
    MPI_Reduce(&stop_clock, &ttime, 1, MPI_DOUBLE, MPI_MAX, 0, mpi_world.comm());
    if (mpi_world.rank() == 0) 
    {
      std::cout << "Total Converge time: " << ttime << std::endl;
      gather.saveToBinaryFile("TEST.bin");
    }
  } else {
    if (mpi_world.rank() == 0) std::cout << "Fail to converge" << std::endl;
  }

  
  return 0;
}