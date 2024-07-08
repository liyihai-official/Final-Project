cmake_minimum_required(VERSION 3.10)

# Project Information
project(FinalProject VERSION 5.0)

# Set MPI as the compiler
find_package(MPI REQUIRED)
if (MPI_FOUND)
    set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
endif()

# C/C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include Directory
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add O3 optimization flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Add OpenMP flag
set(OpenMP_CXX_FLAGS "-fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Executable Files
add_executable(main main.cc)
add_executable(naive_hybird hybrid.cpp)
add_executable(hybird hybrid_2.cpp)
add_executable(demo hybrid_demo.cc)

# Link package
# find_package(SomeLibrary REQUIRED)
# target_link_libraries(MyExecutable SomeLibrary)
