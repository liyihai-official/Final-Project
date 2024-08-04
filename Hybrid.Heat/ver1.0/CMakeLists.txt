cmake_minimum_required(VERSION 3.10)
project(FinalProject VERSION 6.0)

# 查找 MPI 包
find_package(MPI REQUIRED)
if (MPI_FOUND)
  set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
endif()

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 设置包含目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(OpenMP_CXX_FLAGS "-fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# 收集所有源文件
file(GLOB_RECURSE PROJECT_SOURCES
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

# 添加可执行文件，并链接所有源文件
add_executable(test test/test.cc ${PROJECT_SOURCES})
add_executable(main1 main1/main1.cpp ${PROJECT_SOURCES})

# 如果是发布模式，则启用 NDEBUG 宏
if (CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_definitions(test PRIVATE NDEBUG)
  target_compile_definitions(main1 PRIVATE NDEBUG)
endif()

# 链接 MPI 库
target_link_libraries(test ${MPI_LIBRARIES})
target_link_libraries(main1 ${MPI_LIBRARIES})
