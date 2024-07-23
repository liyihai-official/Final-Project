#include <iostream>
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <fdm/heat.hpp>
#include <fdm/evolve.hpp>

#if !defined(NX) || !defined(NY) || !defined(NZ)
#define NX 50+2
#define NY 50+2
#define NZ 50+2
#endif


typedef double value_type;

int main ( int argc, char ** argv )
{
  int num_threads {1};
  constexpr int root_proc {0};
  constexpr value_type tol {1E-3};
  constexpr std::size_t nsteps {100000}, stepinterval {nsteps / 100000}, 
            numDim {3}, nx {NX}, ny {NY}, nz {NZ};

  bool converge {false};

  value_type ldiff {0.0}, gdiff {0.0}, ttime {0.0};

  // Setups
  auto mpi_world {final_project::mpi::env(argc, argv)};
  auto glob_shape {final_project::__detail::__types::__multi_array_shape<numDim>(nx, ny, nz)};
  auto heat_equation {final_project::heat_equation<value_type, numDim>(glob_shape)};

  MPI_Barrier(mpi_world.comm());
  auto gather {final_project::array::array_base<value_type, numDim>(glob_shape)};
  auto ping {final_project::array::array_distribute<value_type, numDim>(glob_shape, mpi_world)};
  auto pong {final_project::array::array_distribute<value_type, numDim>(glob_shape, mpi_world)};

  // Setups
  ping.fill_boundary(10); pong.fill_boundary(10);

  MPI_Barrier(mpi_world.comm());
  sleep(1);
  MPI_Barrier(mpi_world.comm());


  // Setups
  #pragma omp parallel num_threads(2)
  {
    #ifdef _OPENMP
      #pragma omp master
      num_threads = omp_get_num_threads();
    #endif

    int omp_id { omp_get_thread_num() };
    value_type omp_ldiff_bulk {0.0}, omp_ldiff_boundary {0.0};    

    // Brief information of setups
    #pragma omp single
    {
      if (root_proc == mpi_world.rank())
      {
        std::cout << numDim << "Dimension Simulation Parameters: "      << std::endl;
        std::cout << "\tRows: "       << nx 
                  << "\n\tColumns: "  << ny 
                  << "\n\tHeight: "   << nz         << std::endl;
        std::cout << "\tTime steps: " << nsteps     << std::endl;
        std::cout << "\tTolerance: "  << tol        << std::endl;

        std::cout << "MPI Parameters: "             << std::endl;
        std::cout << "\tNumber of MPI Processes: "  << mpi_world.size()   << std::endl;
        std::cout << "\tRoot Process: "             << root_proc          << std::endl;


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
    }

    // Time Evolve
    auto start_clock {MPI_Wtime()};
    for ( int iter = 1; iter <= 2; ++iter )
    {
      ldiff = 0;
      if (converge) { break; }
      omp_ldiff_boundary = update_ping_pong_omp_boundary(ping, pong, heat_equation);
      #pragma omp single
      exchange_ping_pong1(ping);
      omp_ldiff_bulk = update_ping_pong_omp_bulk(ping, pong, heat_equation);
      // #pragma omp barrier

    

      #pragma omp barrier
      #pragma omp master
      {
        std::cout << pong.get_array() << std::endl;
      }
      #pragma omp barrier

      #pragma omp critical
      {
        // if (iter > 8) std::cout << "  >>>> " << omp_ldiff_boundary  << " <<< " <<  omp_ldiff_bulk << std::endl;
        ldiff += omp_ldiff_boundary;
        ldiff += omp_ldiff_bulk;
      }
      #pragma omp barrier

      #pragma omp single
      {
        MPI_Allreduce(&ldiff, &gdiff, 1, MPI_DOUBLE, MPI_SUM, mpi_world.comm());

        if (mpi_world.rank() == root_proc && iter % stepinterval == root_proc) 
          std::cout << std::fixed << std::setprecision(13) << std::setw(15) << gdiff << std::endl;

        if (gdiff <= tol)
        {
          if (mpi_world.rank() == root_proc) 
            std::cout << "Converge at : " 
                      << std::fixed << std::setw(7) << iter
                      << std::endl;
                      
          converge = true;
        }  
        ping.swap(pong); 
      }
    }
    auto stop_clock {MPI_Wtime()};


    // results
    #pragma omp master
    {
      if (converge)
      {
        final_project::array::Gather(gather, ping);
        MPI_Reduce(&stop_clock, &ttime, 1, MPI_DOUBLE, MPI_MAX, 0, mpi_world.comm());

        if ( root_proc == mpi_world.rank())
        {
          std::cout << "Total Converge time: " << ttime << std::endl;
          gather.saveToBinaryFile("TEST.bin");
        }
      } 
      else {
        if ( root_proc == mpi_world.rank() ) std::cout << "Fail to converge" << std::endl;
      }
    }

  } // end of omp parallel

  return 0;
}