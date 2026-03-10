# FindZstd.cmake — locate zstd headers and library
#
# Creates imported target: Zstd::Zstd

include(FindPackageHandleStandardArgs)

find_path(ZSTD_INCLUDE_DIR NAMES zstd.h
    PATH_SUFFIXES include)

find_library(ZSTD_LIBRARY NAMES zstd libzstd_static zstd_static
    PATH_SUFFIXES lib static)

find_package_handle_standard_args(Zstd
    REQUIRED_VARS ZSTD_LIBRARY ZSTD_INCLUDE_DIR)

if(Zstd_FOUND AND NOT TARGET Zstd::Zstd)
    add_library(Zstd::Zstd UNKNOWN IMPORTED)
    set_target_properties(Zstd::Zstd PROPERTIES
        IMPORTED_LOCATION "${ZSTD_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ZSTD_INCLUDE_DIR}"
    )
endif()
