add_executable(test_cuda
    test.c
)

target_link_libraries(test_cuda
    PRIVATE
        transpose
        CUDA::cuda_driver
        CUDA::cudart
)

target_compile_options(test_cuda
    PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -Wpedantic>
)

set_target_properties(test_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    BUILD_RPATH "/run/opengl-driver/lib"
    INSTALL_RPATH "/run/opengl-driver/lib"
)

add_executable(test_ctx_null
    test_ctx_null.c
)

target_link_libraries(test_ctx_null
    PRIVATE
        CUDA::cuda_driver
)

set_target_properties(test_ctx_null PROPERTIES
    BUILD_RPATH "/run/opengl-driver/lib"
    INSTALL_RPATH "/run/opengl-driver/lib"
)

enable_testing()
add_test(NAME test_cuda COMMAND test_cuda)
