cmake_minimum_required(VERSION 3.10)

# Project Infomation
project(FinalProject VERSION 5.0)

# C/C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include Directory
include_directories(${PROJECT_SOURCE_DIR}/include)

# Executable Files
add_executable(main main.cc)

# Link package
# find_package(SomeLibrary REQUIRED)
# target_link_libraries(MyExecutable SomeLibrary)