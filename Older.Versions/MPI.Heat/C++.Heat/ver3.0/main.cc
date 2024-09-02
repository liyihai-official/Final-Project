#include <mpi.h>

#include "final_project.cpp"
#include <chrono>
int main ( int argc ,char ** argv)
{
  /// Optimization tests of std::move, to determine if the function makes copies during calling.
  ///

  // auto A = final_project::array2d<double>(5, 4);
  // auto B = final_project::array2d<double>(4, 3);
  // A.fill(1);
  // B.fill(1);

  // auto start = std::chrono::steady_clock::now();
  // A*B;
  // auto end = std::chrono::steady_clock::now();
  // std::cout 
  // << "Dot Product Elapsed Time: " 
  // << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " micro s\n" 
  // << std::endl;


  // A = final_project::array2d<double>(50, 40);
  // B = final_project::array2d<double>(40, 30);
  // A.fill(1);
  // B.fill(1);

  // start = std::chrono::steady_clock::now();
  // A*B;
  // end = std::chrono::steady_clock::now();
  // std::cout 
  // << "Dot Product Elapsed Time: " 
  // << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " micro s\n" 
  // << std::endl;

  // A = final_project::array2d<double>(500, 400);
  // B = final_project::array2d<double>(400, 300);
  // A.fill(1);
  // B.fill(1);

  // start = std::chrono::steady_clock::now();
  // A*B;
  // end = std::chrono::steady_clock::now();
  // std::cout 
  // << "Dot Product Elapsed Time: " 
  // << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n" 
  // << std::endl;

  /// 
  /// End of Optimization Tests 


  /// The Hybrid Message Passing and Shared Memory Parallelizing Tests 
  int req {MPI_THREAD_MULTIPLE}, prov;
  MPI_Init_thread(&argc, &argv, req, &prov);


  MPI_Finalize();


  return 0;
}