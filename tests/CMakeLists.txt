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

# AVX2 helper library
add_library(div_avx2 STATIC div.avx2.c)
target_compile_options(div_avx2 PRIVATE -mavx2 -Wall -Wextra -Wpedantic)
target_include_directories(div_avx2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# AVX2 test - always build, runtime check will skip if CPU doesn't support
add_my_test(index.ops.avx2 index.ops.avx2.c OpenMP::OpenMP_C)
target_compile_options(index.ops.avx2 PRIVATE -mavx2)
target_link_libraries(index.ops.avx2 PRIVATE div_avx2)

# Magic division test
add_executable(test_magic_div test_magic_div.c)
target_compile_options(test_magic_div PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(test_magic_div PRIVATE div_avx2)

# AVX2 division test
add_executable(test_avx2_div test_avx2_div.c)
target_compile_options(test_avx2_div PRIVATE -mavx2 -Wall -Wextra -Wpedantic)
target_link_libraries(test_avx2_div PRIVATE div_avx2)

