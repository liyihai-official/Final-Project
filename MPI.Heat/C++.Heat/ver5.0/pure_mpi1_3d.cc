#include <iostream>
#include <cmath>
#include <cstring>





#include "fdm/evolve.hpp"
#include "fdm/heat.hpp"

#if !defined(NX) || !defined(NY) || !defined(NZ)
#define NX 50+2
#define NY 50+2
#define NZ 50+2
#endif

typedef double value_type;

int main(int argc, char ** argv)
{
  constexpr int root_proc {0};
  constexpr double tol {1E-3};
  constexpr std::size_t nsteps {100000}, stepinterval {nsteps / 10000};
  constexpr std::size_t numDIM {3}, nx {NX}, ny {NY}, nz {NZ};

  bool converge {false};

  std::size_t iter {0};  
  double gdiff {0.0}, ldiff {0.0}, ttime {0.0};

  // Setups 
  auto mpi_world  {final_project::mpi::env(argc, argv)};
  auto glob_shape {final_project::__detail::__types::__multi_array_shape<numDIM>(nx, ny, nz)};
  auto heat_equation {final_project::heat_equation<double, numDIM>(glob_shape)};

  MPI_Barrier(mpi_world.comm());
  auto gather {final_project::array::array_base<double,3>(glob_shape)};
  auto ping {final_project::array::array_distribute<double, numDIM>(glob_shape, mpi_world)};
  auto pong {final_project::array::array_distribute<double, numDIM>(glob_shape, mpi_world)};

  // setups
  ping.fill_boundary(10);
  pong.fill_boundary(10);

  MPI_Barrier(mpi_world.comm());
  sleep(1);
  MPI_Barrier(mpi_world.comm());










  // Brief information of setups
  if (root_proc == mpi_world.rank())
  {
    std::cout << numDIM << "Dimension Simulation Parameters: " << std::endl;
    std::cout << "\tRows: "       << nx 
              << "\n\tColumns: "  << ny 
              << "\n\tHeight: "   << nz           << std::endl;
    std::cout << "\tTime steps: " << nsteps       << std::endl;
    std::cout << "\tTolerance: "  << tol        << std::endl;

    std::cout << "MPI Parameters: "               << std::endl;
    std::cout << "\tNumber of MPI Processes: "    << mpi_world.size() << std::endl;
    std::cout << "\tRoot Process: "               << root_proc        << std::endl;


    std::cout << "Heat Parameters: "    << std::endl;
    std::cout << "\tCoefficient: "      << heat_equation.coff       << "\n"
              << "\tTime resolution: "  << heat_equation.dt         << "\n"
              << "\tWeights: "          << heat_equation.weights[0] << ", " 
                                        << heat_equation.weights[1] << ", " 
                                        << heat_equation.weights[2] << "\n"
              << "\tdxs: "              << heat_equation.dxs[0]     << ", " 
                                        << heat_equation.dxs[1]     << ", " 
                                        << heat_equation.dxs[2]     << std::endl;
  }






  // Time Evolve
  auto start_clock {MPI_Wtime()};
  for (iter=1; iter <= nsteps; ++iter)
  {
    exchange_ping_pong1(ping);
    ldiff = update_ping_pong1(ping, pong, heat_equation);
    MPI_Allreduce(&ldiff, &gdiff, 1, MPI_DOUBLE, MPI_SUM, mpi_world.comm());

    if (mpi_world.rank() == root_proc && iter % stepinterval == root_proc) 
    {
      std::cout << std::fixed << std::setprecision(13) << std::setw(15) << gdiff << std::endl;
    }

    if (gdiff  <= tol) 
    {
      if (mpi_world.rank() == root_proc) 
        std::cout << "Converge at : " 
                  << std::fixed << std::setw(7) << iter
                  << std::endl;

      converge = true;
      break;
    } else {
      ping.swap(pong);
    }
  }
  auto stop_clock {MPI_Wtime()-start_clock};







  // results
  if (converge)
  {
    Gather(gather, ping);
    MPI_Reduce(&stop_clock, &ttime, 1, MPI_DOUBLE, MPI_MAX, 0, mpi_world.comm());
    if (mpi_world.rank() == root_proc) 
    {
      std::cout << "Total Converge time: " << ttime << std::endl;
      gather.saveToBinaryFile("TEST.bin");
    }
  } 
  else {
    if (mpi_world.rank() == root_proc) std::cout << "Fail to converge" << std::endl;
  }


  return 0;
}