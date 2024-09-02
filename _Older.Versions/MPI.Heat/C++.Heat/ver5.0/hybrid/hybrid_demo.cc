#include <string>
#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <unistd.h>

#ifdef _OPENMP
  #include <omp.h>
#endif
void exchange(const int iter, const int id)
{
  sleep(2);
  std::cout << "Thread: " << id
            << " Exchange at " 
            << iter << " iteration. " << std::endl;
}

void evolve(const int iter)
{
  sleep(2);
  std::cout << "Evolving at "
            << iter << " iteration. " << std::endl;
}

void swap(const int iter)
{
  sleep(0.5);
}

int main( int argc, char ** argv)
{

  const int nsteps {10};

  int provided;     // MPI Thread support level

  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  if (provided < MPI_THREAD_SERIALIZED)
  {
    std::cerr << "MPI_THREAD_SERIALIZED thread support level required." << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 5);
  }

  int num_threads {1};

  #pragma omp parallel num_threads(2)
  {
    const int id {omp_get_thread_num()};
    #ifdef _OPENMP
      #pragma omp master 
        num_threads = omp_get_num_threads();
    #endif 


    #pragma omp single
    {
      int rank, size;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      if (rank == 0) 
      {
        std::cout << "Simulation parameters: "
                  << "ROWS: " << 0 << " COLS: " << 0
                  << " Time Steps: " << nsteps << std::endl;
        
        std::cout << "Number of MPI Tasks: " << size << std::endl;
        std::cout << "Number of OpenMP Threads: " << num_threads << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Average Temperature at start: " << 0 << std::endl;
      }
    }

    auto start_clock {MPI_Wtime()};
    for (int iter {1}; iter <= nsteps; ++iter)
    {
      #pragma omp single
      exchange(iter, id);
      evolve(iter);
      #pragma omp single
      {
        if (iter & 10 == 0)
        {
          std::cout << "Write Image to File (Parallel I/O)." << std::endl;
        }

        swap(iter);
      } // end omp single
      
    }
    auto stop_clock {MPI_Wtime()};

    #pragma omp master
    {
      auto average_temp {0};
      int rank, size;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      if (rank == 0) 
      { 
        std::cout << "Iteration took " << (stop_clock - start_clock)
                  << " seconds." << std::endl;
        
        std::cout << "Average temperature: " << average_temp << std::endl;

        if (1 == argc) 
        {

        }
      }
       
    } // end of omp master

  } // end of omp parallel


  // Output the final field
  sleep(2);
  std::cout << "Write Image to File Finally (Parallel I/O)." << std::endl;



  MPI_Finalize();






  return 0;
}