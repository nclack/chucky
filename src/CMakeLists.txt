if(MSVC)
    set(C_WARN_FLAGS /W3)
    set(CUDA_HOST_WARN_FLAGS --compiler-options=/W3)
else()
    set(C_WARN_FLAGS -Wall -Wextra -Wpedantic)
    set(CUDA_HOST_WARN_FLAGS --compiler-options=-Wall,-Wextra)
endif()

add_library(transpose STATIC
    transpose.cu
    transpose.h
)

target_include_directories(transpose
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(transpose PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(transpose
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_HOST_WARN_FLAGS}>
)

# nvcomp — find_package(nvcomp) is broken on Nix (split outputs),
# so locate headers and library manually.
find_path(NVCOMP_INCLUDE_DIR NAMES nvcomp.h
    PATH_SUFFIXES include REQUIRED)
find_library(NVCOMP_LIBRARY NAMES nvcomp_static nvcomp
    PATH_SUFFIXES lib lib/${CUDAToolkit_VERSION_MAJOR} REQUIRED)

add_library(nvcomp::nvcomp UNKNOWN IMPORTED)
set_target_properties(nvcomp::nvcomp PROPERTIES
    IMPORTED_LOCATION "${NVCOMP_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${NVCOMP_INCLUDE_DIR}"
)
if(NVCOMP_LIBRARY MATCHES "nvcomp_static")
    set_property(TARGET nvcomp::nvcomp PROPERTY
        INTERFACE_COMPILE_DEFINITIONS NVCOMP_STATIC_DEFINE)
endif()

add_library(chucky_log STATIC
    log/log.c
    log/log.h
)

target_include_directories(chucky_log
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(chucky_log PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(chucky_log
    PRIVATE
        $<$<COMPILE_LANGUAGE:C>:${C_WARN_FLAGS}>
)

add_library(compress STATIC
    compress.cu
    compress.h
)

target_include_directories(compress
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(compress
    PUBLIC
        nvcomp::nvcomp
        CUDA::cudart_static
        chucky_log
)

set_target_properties(compress PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(compress
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_HOST_WARN_FLAGS}>
)

add_library(aggregate STATIC
    aggregate.cu
    aggregate.h
)

target_include_directories(aggregate
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(aggregate
    PUBLIC
        CUDA::cudart_static
        CUDA::cuda_driver
        chucky_log
)

set_target_properties(aggregate PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(aggregate
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_HOST_WARN_FLAGS}>
)

if(WIN32)
    set(PLATFORM_SRC platform.win32.c)
else()
    set(PLATFORM_SRC platform.posix.c)
endif()

add_library(stream STATIC
    stream.c
    stream.h
    ${PLATFORM_SRC}
    platform.h
)

target_include_directories(stream
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(stream
    PUBLIC
        transpose
        compress
        aggregate
        chucky_log
        CUDA::cuda_driver
)

set_target_properties(stream PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(stream
    PRIVATE
        $<$<COMPILE_LANGUAGE:C>:${C_WARN_FLAGS}>
)
