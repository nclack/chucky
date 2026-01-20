# Coding Style Guide

## File Organization

### Header Files (.hh)

- **Public headers** (`*.hh`) should contain **interface declarations only**
- No implementation details in public headers
- Keep public API surface minimal and clean

### Private Headers (.priv.hh)

- **Private headers** (`*.priv.hh`) contain implementation details that must be in header files
- Use when:
  - Template implementations need to be visible
  - Private helper classes/structs are needed by implementation
  - Internal APIs that shouldn't be public

### Implementation Files

- **C++ implementation** (`*.cc`) - standard C++ implementation code
- **CUDA implementation** (`*.cu`) - CUDA kernel and device code
- All implementation logic goes here, not in public headers

## Error Handling

- Use **return codes** for error propagation (not exceptions)
- Use **spdlog** for logging errors and diagnostics
- Mark functions `noexcept` where possible for performance

## Writer Interface

- Writers accept `std::span<const std::byte>` as input
- Return `WriteResult` containing unconsumed bytes on success
- Empty span indicates all bytes were consumed
- Caller retries unconsumed bytes later
