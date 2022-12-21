cmake_minimum_required(VERSION 3.24)
project(CodingProject3 CUDA CXX)


set(CMAKE_CUDA_STANDARD 14)

add_executable(vecadd_cpu
        vecadd_cpu.cpp)

add_executable(vecadd_gpu_1t
        vecadd_gpu_1t.cu)
set_target_properties(vecadd_gpu_1t
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(vecadd_gpu_256t
        vecadd_gpu_256t.cu)
set_target_properties(vecadd_gpu_256t
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(vecadd_gpu_256t_manyblocks
        vecadd_gpu_256t_manyblocks.cu)
set_target_properties(vecadd_gpu_256t_manyblocks
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(vecadd_gpu_256t_manyblocks_prefetch
        vecadd_gpu_256t_manyblocks_prefetch.cu)
set_target_properties(vecadd_gpu_256t_manyblocks_prefetch
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)




