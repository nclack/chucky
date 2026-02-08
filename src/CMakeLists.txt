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
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(transpose
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall,-Wextra>
)

add_library(stream STATIC
    stream.c
    stream.h
    log/log.c
    log/log.h
)

target_include_directories(stream
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(stream
    PUBLIC
        transpose
        CUDA::cuda_driver
)

set_target_properties(stream PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(stream
    PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -Wpedantic>
)
