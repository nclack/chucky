# OpenMP.cmake — add OpenMP support to a target
#
# enable_openmp(target ...)
#   Adds OpenMP compile flags and link libraries to each target.

if(MSVC)
    set(_OPENMP_FLAGS /openmp:llvm)
else()
    find_package(OpenMP REQUIRED COMPONENTS C)
    set(_OPENMP_FLAGS ${OpenMP_C_FLAGS})
endif()

function(enable_openmp)
    foreach(tgt IN LISTS ARGN)
        target_compile_options(${tgt} PRIVATE ${_OPENMP_FLAGS})
        if(NOT MSVC)
            target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_C)
        endif()
    endforeach()
endfunction()
