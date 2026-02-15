cmake_minimum_required(VERSION 3.18)
project(chucky LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_ARCHITECTURES 100)

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:C>:/experimental:c11atomics>)
    # nvcomp_static and zstd_static are built with /MD; cudart_static uses /MT.
    # Stay on /MD to match the majority; suppress the harmless LNK4098 from
    # cudart_static's LIBCMT reference.
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    add_link_options(/NODEFAULTLIB:libcmt)
endif()

find_package(CUDAToolkit REQUIRED)

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
