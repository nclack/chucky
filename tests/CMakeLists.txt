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

# Index operations utility library (shared between scalar and AVX2 tests)
add_library(index_ops_util STATIC index.ops.util.c)
target_compile_options(index_ops_util PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(index_ops_util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_my_test(index.ops index.ops.c OpenMP::OpenMP_C)
target_link_libraries(index.ops PRIVATE index_ops_util stream)

# AVX2 helper library
add_library(div_avx2 STATIC div.avx2.c)
target_compile_options(div_avx2 PRIVATE -mavx2 -Wall -Wextra -Wpedantic)
target_include_directories(div_avx2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# AVX2 test - always build, runtime check will skip if CPU doesn't support
add_my_test(index.ops.avx2 index.ops.avx2.c OpenMP::OpenMP_C)
target_compile_options(index.ops.avx2 PRIVATE -mavx2)
target_link_libraries(index.ops.avx2 PRIVATE div_avx2 index_ops_util stream)

# Magic division test
add_executable(test_magic_div test_magic_div.c)
target_compile_options(test_magic_div PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(test_magic_div PRIVATE div_avx2)

# AVX2 division test
add_executable(test_avx2_div test_avx2_div.c)
target_compile_options(test_avx2_div PRIVATE -mavx2 -Wall -Wextra -Wpedantic)
target_link_libraries(test_avx2_div PRIVATE div_avx2)

# Transpose indices CUDA test
add_my_test(index.ops.cuda index.ops.cuda.c transpose index_ops_util CUDA::cudart)

# Streaming tiled transpose test
find_path(ZSTD_INCLUDE_DIR NAMES zstd.h REQUIRED)
find_library(ZSTD_LIBRARY NAMES zstd REQUIRED)

add_my_test(test_stream test_stream.c stream CUDA::cudart)
target_link_libraries(test_stream PRIVATE index_ops_util ${ZSTD_LIBRARY})
target_include_directories(test_stream PRIVATE ${ZSTD_INCLUDE_DIR})
target_sources(test_stream PRIVATE writer.mem.c)

add_my_test(test_bench test_bench.c stream CUDA::cudart)
target_link_libraries(test_bench PRIVATE index_ops_util ${ZSTD_LIBRARY})
target_include_directories(test_bench PRIVATE ${ZSTD_INCLUDE_DIR})

