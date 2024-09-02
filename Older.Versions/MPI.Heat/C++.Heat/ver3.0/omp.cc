// #include <mpi.h>
// #include <omp.h>

// #include "final_project.cpp"

// #define MAX_N_X 13+2
// #define MAX_N_Y 13+2

// int main (int argc, char ** argv)
// { 
//   int required = MPI_THREAD_MULTIPLE;  // 请求 MPI_THREAD_MULTIPLE 级别
//   int provided;

//   auto world = mpi::env(argc, argv, required, &provided);

//   int i;
//   double loc_diff, glob_diff {10}, t1, t2;
//   constexpr int reorder {1}, dimension {2}, root {0};

//   int dims[dimension], coords[dimension], periods[dimension];
//   for (short int i = 0; i < dimension; ++i) {dims[i] = 0; periods[i] = 0;}

//   /* MPI Cartesian Inits */
//   MPI_Comm comm_cart;
//   MPI_Dims_create(world.size(), dimension, dims);
//   MPI_Cart_create(MPI_COMM_WORLD, dimension, dims, periods, reorder, &comm_cart);

//   final_project::array2d_distribute<double> a, b;
//   final_project::array2d<double> gather (MAX_N_X, MAX_N_Y);

// #pragma omp parallel num_threads(2) private(i) 
// {
//   MPI_Datatype vecs_omp[2];
//   int omp_pid = omp_get_thread_num();
//   a.distribute(MAX_N_X, MAX_N_Y, dims, omp_pid, vecs_omp, comm_cart);

// #pragma omp barrier
//   b.distribute(MAX_N_X, MAX_N_Y, dims, omp_pid, vecs_omp, comm_cart);


// // #pragma omp single
// {
//   init_conditions_heat2d(a, b);
//   init_conditions_heat2d(gather);

//   a.sweep_setup_heat2d(1, 1);
//   b.sweep_setup_heat2d(1, 1);
// }

//   t1 = MPI_Wtime();
//     for (i = 0; i < 100; ++i )
//     {
//       // a.SR_OMP_exchange2d(omp_pid, vecs_omp);
//       a.sweep_heat2d_omp1(b, omp_pid);
// // // #pragma omp barrier

// #pragma omp critical
// {
//       b.SR_OMP_exchange2d(omp_pid, vecs_omp);
// }
// // // #pragma omp barrier

//       b.sweep_heat2d_omp1(a, omp_pid);
// // // #pragma omp barrier
// #pragma omp critical
// {
//       a.SR_OMP_exchange2d(omp_pid, vecs_omp);
// }
// // #pragma omp barrier      
//     }
//   t2 = MPI_Wtime();

//   // if (omp_pid == 0)
//   {

//     // a.SR_OMP_exchange2d(omp_pid, vecs_omp);
//     // a.sweep_heat2d_omp1(b, omp_pid);
//     // b.SR_OMP_exchange2d(omp_pid, vecs_omp);
//     // b.sweep_heat2d_omp1(a, omp_pid);
//   }

// }


//   final_project::print_in_order(a);

//   // MPI_Barrier(comm_cart);
  


//   return 0;
// }
