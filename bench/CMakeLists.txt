if(MSVC)
    set(C_WARN_FLAGS /W3)
else()
    set(C_WARN_FLAGS -Wall -Wextra -Wpedantic)
endif()

find_path(ZSTD_INCLUDE_DIR NAMES zstd.h
    PATH_SUFFIXES include REQUIRED)
find_library(ZSTD_LIBRARY NAMES zstd libzstd_static zstd_static
    PATH_SUFFIXES lib static REQUIRED)

# bench_util: static library with pipeline benchmark infrastructure
add_library(bench_util STATIC bench_util.c bench_util.h)
target_include_directories(bench_util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bench_util PUBLIC stream zarr_sink test_data CUDA::cuda_driver)
target_compile_options(bench_util PRIVATE $<$<COMPILE_LANGUAGE:C>:${C_WARN_FLAGS}>)
set_target_properties(bench_util PROPERTIES POSITION_INDEPENDENT_CODE ON)

# bench_stream: per-scenario, per-mode executables
set(BENCH_STREAM_SOURCES
    bench_stream_256cube_single.c
    bench_stream_256cube_multiscale.c
    bench_stream_256cube_multiscale_dim0.c
    bench_stream_orca2_single.c
    bench_stream_orca2_multiscale.c
    bench_stream_orca2_multiscale_dim0.c
)

foreach(src IN LISTS BENCH_STREAM_SOURCES)
    string(REPLACE ".c" "" target ${src})
    add_executable(${target} ${src})
    target_link_libraries(${target} PRIVATE bench_util CUDA::cudart_static ${ZSTD_LIBRARY})
    target_include_directories(${target} PRIVATE ${ZSTD_INCLUDE_DIR})
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:C>:${C_WARN_FLAGS}>)
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(NOT WIN32)
        set_target_properties(${target} PROPERTIES
            BUILD_RPATH "/run/opengl-driver/lib"
            INSTALL_RPATH "/run/opengl-driver/lib"
        )
    endif()
endforeach()

# bench_scatter_lut: scatter LUT kernel benchmark (independent of bench_util)
add_executable(bench_scatter_lut bench_scatter_lut.c)
target_link_libraries(bench_scatter_lut PRIVATE lod lod_plan index_ops CUDA::cuda_driver CUDA::cudart_static)
target_compile_options(bench_scatter_lut PRIVATE $<$<COMPILE_LANGUAGE:C>:${C_WARN_FLAGS}>)
set_target_properties(bench_scatter_lut PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT WIN32)
    set_target_properties(bench_scatter_lut PROPERTIES
        BUILD_RPATH "/run/opengl-driver/lib"
        INSTALL_RPATH "/run/opengl-driver/lib"
    )
endif()
