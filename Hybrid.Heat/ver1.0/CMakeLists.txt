cmake_minimum_required(VERSION 3.10)

# 项目信息
project(FinalProject VERSION 6.0)

# 查找 Torch 包
find_package(Torch REQUIRED)

# 查找 MPI 包
find_package(MPI REQUIRED)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 设置包含目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 设置编译选项
set(OpenMP_CXX_FLAGS "-fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3")

# 如果找到了 MPI 包，设置编译器
if (MPI_FOUND)
  set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
endif()

# 收集所有源文件
file(GLOB_RECURSE PROJECT_SOURCES
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

# 添加可执行文件，并链接所有源文件
add_executable(main2 main2/main2.cpp ${PROJECT_SOURCES})
add_executable(main1 main1/main1.cpp ${PROJECT_SOURCES})
add_executable(main1_3d main1/main1_3d.cpp ${PROJECT_SOURCES})

# 链接 Torch 库
target_link_libraries(main2 "${TORCH_LIBRARIES}")

# 链接 MPI 库
target_link_libraries(main1 ${MPI_LIBRARIES})
target_link_libraries(main1_3d ${MPI_LIBRARIES})

# 添加 O3 优化标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Windows 上的特定配置
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET main2
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:main2>)
endif (MSVC)

# 如果是发布模式，则启用 NDEBUG 宏
if (CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_definitions(main2 PRIVATE NDEBUG)
  target_compile_definitions(main1 PRIVATE NDEBUG)
  target_compile_definitions(main1_3d PRIVATE NDEBUG)
endif()
