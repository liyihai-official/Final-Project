cmake_minimum_required(VERSION 3.10)

# Project Information
project(FinalProject VERSION 5.0)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(main main.cc)
add_executable(pinn_2d pinn_2d.cpp)
add_executable(example minist.cpp)

target_link_libraries(main "${TORCH_LIBRARIES}")
target_link_libraries(pinn_2d "${TORCH_LIBRARIES}")
target_link_libraries(example "${TORCH_LIBRARIES}")

set_property(TARGET main PROPERTY CXX_STANDARD 20)
set_property(TARGET pinn_2d PROPERTY CXX_STANDARD 20)
set_property(TARGET example PROPERTY CXX_STANDARD 20)

# Add O3 optimization flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET main
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:main>)
  add_custom_command(TARGET pinn_2d
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${TORCH_DLLS}
                    $<TARGET_FILE_DIR:pinn_2d)

  add_custom_command(TARGET example
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND}
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example>)


endif (MSVC)