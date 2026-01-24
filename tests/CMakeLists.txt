# Helper function to create test executables with common settings
function(add_my_test TGT SOURCE_FILE)
    add_executable(${TGT} ${SOURCE_FILE})

    target_link_libraries(${TGT}
        PRIVATE
            CUDA::cuda_driver
            ${ARGN}
    )

    set_target_properties(${TGT} PROPERTIES
        BUILD_RPATH "/run/opengl-driver/lib"
        INSTALL_RPATH "/run/opengl-driver/lib"
    )

    target_compile_options(${TGT}
        PRIVATE
            $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -Wpedantic>
    )

    set_target_properties(${TGT} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
    
    add_test(NAME "test-${TGT}" COMMAND ${TGT})
endfunction()

add_my_test(test_cuda test.c transpose CUDA::cudart)
add_my_test(test_ctx_null test_ctx_null.c)

find_package(OpenMP REQUIRED)
add_my_test(index.ops index.ops.c OpenMP::OpenMP_C)

