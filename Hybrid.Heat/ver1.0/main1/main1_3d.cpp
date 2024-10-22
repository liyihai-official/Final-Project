///
/// @file main1_3d.cpp
/// 
/// @brief
///
///
///
/// @author LI Yihai
///
///
///

/// Message Passing Interface & OpenMP 
#include <mpi.h>
#include <omp.h>

/// Standard Library Files
#include <cmath>
#include <numbers>
#include <iostream>
#include <functional>

/// Final Project header files
#include <types.hpp>
#include <helper.hpp>
/// PDE 
#include <pde/detials/Heat_3D.hpp>
#include <pde/detials/InitializationsC.hpp>
#include <pde/detials/BoundaryConditions/BoundaryConditions_3D.hpp>

/// Problem Size + Boundaries
#if !defined(NX) || !defined (NY) || !defined(NZ)
#define NX 200+2
#define NY 200+2
#define NZ 200+2
#endif

/// Using datatypes
using Integer = final_project::Integer;
using Char    = final_project::Char;
using Double  = final_project::Double;
using Float   = final_project::Float;
using size_type = final_project::Dworld;

using maintype  = Float;

using BCFunction = std::function<maintype(maintype, maintype, maintype, maintype)>;
using ICFunction = std::function<maintype(maintype, maintype, maintype)>;


Integer 
  main ( Integer argc, Char ** argv )
{
  Integer opt {0}, iter {0};
  final_project::Strategy strategy {final_project::Strategy::UNKNOWN};
  final_project::String filename {""};        // default empty filename (not saving results).

  constexpr Integer root_proc {0};
  constexpr maintype tol {1E-8};
  constexpr size_type nsteps {100'000'000};
  constexpr size_type numDim {3}, nx {NX}, ny {NY}, nz {NZ};

  auto mpi_world {final_project::mpi::environment(argc, argv)};

  #pragma omp parallel  // Predefined Arguments
  {
    #pragma omp master
    {
if (mpi_world.rank() == 0)
{
  std::cout << "Problem size: " 
            << "\n\tDepth: "     << nx-2 
            << "\n\tColumns: "   << ny-2       
            << "\n\tRows: "      << nz-2      << std::endl;
  std::cout << "MPI Parameters: "             << std::endl;
  std::cout << "\tNumber of MPI Processes: "  << mpi_world.size()  << std::endl;
  std::cout << "OpenMP Threads: " << omp_get_num_threads() << "\n" << std::endl;
}
    }
  }

  while ((opt = getopt(argc, argv, "HhVvF:f:S:s:")) != -1)  // Command Line Arguments
  {
switch (opt) 
{
  case 'S': case 's':
    strategy = final_project::getStrategyfromString(optarg);
    break;
  case 'F': case 'f':
    filename = optarg;
    if (mpi_world.rank() == 0)
    std::cout 
      << "This program will store results to the file: " 
      << filename << std::endl;
    break;
  case 'H': case 'h':
    final_project::helper_message(mpi_world);
    exit(EXIT_FAILURE);
  case 'V': case 'v':
    final_project::version_message(mpi_world);
    exit(EXIT_FAILURE);
  default:
    std::cerr
      << "Invalid option: -" 
      << static_cast<Char>(opt) 
      << "\n";
    exit(EXIT_FAILURE);
}
  }

  /// Heat Equation Object
  final_project::pde::Heat_3D<maintype> obj (mpi_world, nx, ny, nz);

  /// Initial Condition
  ICFunction InitCond { [](maintype x, maintype y, maintype z) { return 0; }};
  final_project::pde::InitialConditions::Init_3D<maintype> IC (InitCond);
  obj.SetHeatInitC(IC);

  /// Boundary Conditions
  final_project::pde::BoundaryConditions_3D<maintype> BC (true, true, true, true, true, true);
  BCFunction Dim000 {[](maintype x, maintype y, maintype z, maintype t){ return y + z - 2 * y * z; }};
  BCFunction Dim001 {[](maintype x, maintype y, maintype z, maintype t){ return 1 - y - z + 2 * y * z; }};

  BCFunction Dim010 {[](maintype x, maintype y, maintype z, maintype t){ return x + z - 2 * x * z; }};
  BCFunction Dim011 {[](maintype x, maintype y, maintype z, maintype t){ return 1 - x - z + 2 * x * z; }};

  BCFunction Dim100 {[](maintype x, maintype y, maintype z, maintype t){ return x + y - 2 * x * y; }};
  BCFunction Dim101 {[](maintype x, maintype y, maintype z, maintype t){ return 1 - x - y + 2 * x * y; }};
  obj.SetHeatBC(BC, Dim000, Dim001, Dim010, Dim011, Dim100, Dim101);

  // std::cout << "\n\n" << std::endl;

  /// Solving with Specified Strategy
  switch (strategy)
  {
    case final_project::Strategy::PURE_MPI:
      iter = obj.solve_pure_mpi(tol, nsteps, root_proc);
      break;
    case final_project::Strategy::HYBRID_0:
      iter = obj.solve_hybrid_mpi_omp(tol, nsteps, root_proc);
      break;
    case final_project::Strategy::HYBRID_1:
      iter = obj.solve_hybrid2_mpi_omp(tol, nsteps, root_proc);
      break;
    default:
      if (mpi_world.rank() == 0)
      {
std::cerr << "Undefined Parallel Strategy, FAIL TO SOLVE. Print Usage Message. MPI_Abort next.\n" << std::endl;
final_project::helper_message(mpi_world);
      }
      FINAL_PROJECT_MPI_ASSERT_GLOBAL(strategy == final_project::Strategy::UNKNOWN);
      break;
  }

  /// Save if needs
  if (!filename.empty()) obj.SaveToBinary(filename);
  // MPI_Barrier(MPI_COMM_WORLD);
  // if (mpi_world.rank() == 0)
  // std::cout << obj.gather << std::endl;
  // final_project::mpi::MPI_SaveToBinary(obj.in, "test.bin");
  
  obj.reset();



  // std::cout << "\n" << std::endl;
  // final_project::multi_array::__detail::__multi_array_shape<2> shape(22,22), shape2(5,5), shape3(4,3);
  // std::cout << shape.strides[0] << ", " << shape.strides[1] << std::endl;
  // final_project::multi_array::__detail::__multi_array_shape<4> shape(6,5,4,3);
  // std::cout << shape.strides[0] <<  ", " << shape.strides[1] <<  ", " <<  shape.strides[2] << ", " << shape.strides[3] << std::endl;

  // std::cout << (shape != shape2) << std::endl;
  // final_project::multi_array::__detail::__multi_array_shape<2> other;
  // other = std::move(shape);
  // std::cout << other[0] << ", " << other[1] << std::endl;


  // final_project::multi_array::__detail::__array<Double, 4> array (shape);//, other (shape2);
  // std::array<Integer, 4> indexes {1,0,0,0};
  // Integer idx {array.get_flat_index(indexes)};
  // // std::cout << &array[array.get_flat_index(indexes)] << "\t" << array.begin() + 60 << "\t" << &array(9,9,9,9) << "\n" << array[-1] << std::endl;
  // std::cout << array(9,9,9,9) << std::endl;


  // array.fill(1); other.fill(2);

  // final_project::multi_array::__detail::__array<Double, 2> move = std::move(array);

  // std::cout << move << std::endl;

  // std::cout << move.__shape[0] << ", " << move.__shape[1] << std::endl;
  // std::cout << array.__shape[0] << std::endl;
  // std::cout << array << "\n" << other << std::endl;
  // array.swap(other);
  // std::cout << array << std::endl;
  // std::cout << array << "\n" << other << std::endl;

  // std::array<Integer, 3> Ns, starts_cpy, array_sizes, array_subsizes, array_starts, indexes;
  // // 定义全局三维数组的大小
  // array_sizes = {16, 18, 19};
  // // int gsizes[3] = {16, 18, 19}; // 全局数组大小为 16x16x16
  // // // 定义数据在每个维度的分布方式
  // // int distribs[3] = {MPI_DISTRIBUTE_BLOCK, MPI_DISTRIBUTE_BLOCK, MPI_DISTRIBUTE_BLOCK};
  // // // 默认块大小
  // // int dargs[3] = {MPI_DISTRIBUTE_DFLT_DARG, MPI_DISTRIBUTE_DFLT_DARG, MPI_DISTRIBUTE_DFLT_DARG};
  // // 每个维度上用于分布的进程数
  
  // int psizes[3] = {2, 2, 2}; // 2x2x2的进程网格
  // MPI_Datatype dtype;

  // // 创建三维分布式数组类型
  // MPI_Type_create_darray(mpi_world.size(), mpi_world.rank(), 3, gsizes, distribs, dargs, psizes, MPI_ORDER_C, MPI_DOUBLE, &dtype);
  // MPI_Type_commit(&dtype);

  // // 获取本地数组的大小
  // int local_size;
  // MPI_Type_size(dtype, &local_size);
  // local_size /= sizeof(double); // 得到的是字节数，除以 sizeof(double) 转化为元素个数

  // // 分配本地数组
  // double *local_array = new double[local_size];
  
  // // 初始化本地数组
  // for (int i = 0; i < local_size; i++) {
  //     local_array[i] = mpi_world.rank() + 1; // 每个进程填充不同的值
  // }

  // // 输出每个进程上的本地数组信息
  // std::cout << "Process " << mpi_world.rank() << " has local array of size " << local_size / 8 << std::endl;

  // 进行一些计算或通信操作...



  // std::cout << shape.dims[0] << std::endl;


  // 释放资源
  // MPI_Type_free(&dtype);
  // delete[] local_array;


  
  return 0;
}