cmake_minimum_required(VERSION 3.10)
project(FinalProject VERSION 6.0)

find_package(MPI REQUIRED)
if (MPI_FOUND)
  set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories($(PROJECT_SOURCE_DIR)/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(OpenMP_CXX_FLAGS "-fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")


add_executable(test test/test.cc)


