cmake_minimum_required(VERSION 3.20)
project(ninfer_cpu_nvfp4_expert_pair LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Threads REQUIRED)

add_executable(ninfer_cpu_nvfp4_expert_pair main.cpp)
target_compile_features(ninfer_cpu_nvfp4_expert_pair PRIVATE cxx_std_20)
target_link_libraries(ninfer_cpu_nvfp4_expert_pair PRIVATE Threads::Threads)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(ninfer_cpu_nvfp4_expert_pair PRIVATE
    -O3
    -mavx2
    -mfma
    -Wall
    -Wextra
    -Wpedantic)
else()
  message(FATAL_ERROR "The Phase-1 CPU probe currently requires GCC or Clang on x86-64 Linux")
endif()
