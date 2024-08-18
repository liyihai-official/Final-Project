cmake_minimum_required(VERSION 3.10)
project(MAP55640_Final_Project VERSION 6.0)

set(PROJECT_AUTHOR "LI Yihai")

# Find Libtorch and MPI
find_package(Torch REQUIRED)
find_package(MPI REQUIRED)
if (MPI_FOUND)
  set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
endif()

# C++ Standard Required, C++ 20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# include path
include_directories(${PROJECT_SOURCE_DIR}/include)

# Compile Flags
set(OpenMP_CXX_FLAGS "-fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3")


# Collect Source Files
file(GLOB_RECURSE PROJECT_SOURCES
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

# Add executable files and source files
add_executable(main1 main1/main1.cpp ${PROJECT_SOURCES})
add_executable(main1_3d main1/main1_3d.cpp ${PROJECT_SOURCES})
add_executable(main2 main2/main2.cpp ${PROJECT_SOURCES})


# Linking to MPI
target_link_libraries(main1 ${MPI_LIBRARIES})
target_link_libraries(main1_3d ${MPI_LIBRARIES})

# Linking to Libtorch
target_link_libraries(main2 "${TORCH_LIBRARIES}")

# Add Optimization flag 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET main
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:main>)
endif (MSVC)

set(NX 100 CACHE STRING "Grid dimension NX")
set(NY 100 CACHE STRING "Grid dimension NY")
set(NZ 50 CACHE STRING "Grid dimension NY")


# Release mode, set to No Debug
if (CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_definitions(main1 PRIVATE NDEBUG NX=${NX} NY=${NY})
  target_compile_definitions(main1_3d PRIVATE NDEBUG NX=${NX} NY=${NY} NZ=${NZ})
  target_compile_definitions(main2 PRIVATE NDEBUG)
endif()

configure_file(config.h.in config.h)