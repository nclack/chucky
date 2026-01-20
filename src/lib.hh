#pragma once

#include <cstdint>
#include <cstddef>

namespace chucky {

/// Maximum number of dimensions supported
constexpr size_t MAX_DIMS = 64;

/// Supported data types
enum class DType : uint8_t {
    U8,   // uint8_t
    U16,  // uint16_t
    U32,  // uint32_t
    F32,  // float
    F64,  // double
};

/// Dimension metadata
struct Dimension {
    uint32_t size_px;       // Size of this dimension in elements
    uint32_t tile_size_px;  // Tile size for this dimension in elements
};

/// Array layout and tiling information
struct Layout {
    uint8_t rank;                    // Number of dimensions
    DType dtype;                     // Element data type
    uint32_t dims[MAX_DIMS];         // Per-dimension info
};

} // namespace chucky
