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
