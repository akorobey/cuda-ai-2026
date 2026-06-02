
project(dk_gelu_cpu)

cmake_minimum_required(VERSION 3.10)

find_package(OpenMP REQUIRED)

add_executable(
    dk_gelu_cpu main.cpp gelu_omp.cpp gelu_omp.h
)

set_property(
    TARGET dk_gelu_cpu PROPERTY CXX_STANDARD 17
)
target_link_libraries(
    dk_gelu_cpu PRIVATE OpenMP::OpenMP_CXX
)
target_include_directories(
    dk_gelu_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_options(
    dk_gelu_cpu PRIVATE -O2 -march=native
)